#!/usr/bin/env python3
"""
Nova Memory System - Query Execution Engine
High-performance execution engine with parallel processing and monitoring
"""

import json
import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
import traceback

from memory_query_optimizer import (
    QueryPlan, ExecutionStatistics, OptimizationContext, MemoryQueryOptimizer
)

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Query execution status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExecutionMode(Enum):
    """Query execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"

@dataclass
class ExecutionContext:
    """Context for query execution"""
    execution_id: str
    nova_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    priority: int = 1
    timeout_seconds: Optional[float] = None
    trace_execution: bool = False
    memory_limit: Optional[int] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ExecutionResult:
    """Result of query execution"""
    execution_id: str
    status: ExecutionStatus
    data: Any = None
    error: Optional[str] = None
    execution_stats: Optional[ExecutionStatistics] = None
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total execution time"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

@dataclass
class OperationResult:
    """Result of individual operation execution"""
    operation_id: str
    operation_type: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    rows_processed: int = 0
    memory_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExecutionMonitor:
    """Monitor and track query executions"""
    
    def __init__(self):
        self.active_executions = {}
        self.execution_history = []
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'peak_memory_usage': 0,
            'total_rows_processed': 0
        }
        self._lock = threading.RLock()
    
    def start_execution(self, execution_id: str, plan: QueryPlan, context: ExecutionContext):
        """Start monitoring an execution"""
        with self._lock:
            self.active_executions[execution_id] = {
                'plan': plan,
                'context': context,
                'started_at': datetime.utcnow(),
                'status': ExecutionStatus.RUNNING
            }
            self.performance_metrics['total_executions'] += 1
    
    def complete_execution(self, execution_id: str, result: ExecutionResult):
        """Complete monitoring an execution"""
        with self._lock:
            if execution_id in self.active_executions:
                execution_info = self.active_executions.pop(execution_id)
                
                # Update metrics
                if result.status == ExecutionStatus.COMPLETED:
                    self.performance_metrics['successful_executions'] += 1
                else:
                    self.performance_metrics['failed_executions'] += 1
                
                if result.execution_time:
                    current_avg = self.performance_metrics['avg_execution_time']
                    total = self.performance_metrics['total_executions']
                    new_avg = ((current_avg * (total - 1)) + result.execution_time) / total
                    self.performance_metrics['avg_execution_time'] = new_avg
                
                if result.execution_stats:
                    self.performance_metrics['peak_memory_usage'] = max(
                        self.performance_metrics['peak_memory_usage'],
                        result.execution_stats.memory_usage
                    )
                    self.performance_metrics['total_rows_processed'] += result.execution_stats.rows_processed
                
                # Add to history
                self.execution_history.append({
                    'execution_id': execution_id,
                    'result': result,
                    'execution_info': execution_info,
                    'completed_at': datetime.utcnow()
                })
                
                # Limit history size
                if len(self.execution_history) > 10000:
                    self.execution_history = self.execution_history[-5000:]
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get currently active executions"""
        with self._lock:
            return [
                {
                    'execution_id': exec_id,
                    'plan_id': info['plan'].plan_id,
                    'nova_id': info['context'].nova_id,
                    'started_at': info['started_at'],
                    'duration': (datetime.utcnow() - info['started_at']).total_seconds()
                }
                for exec_id, info in self.active_executions.items()
            ]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            success_rate = (
                self.performance_metrics['successful_executions'] / 
                max(self.performance_metrics['total_executions'], 1)
            )
            return {
                **self.performance_metrics,
                'success_rate': success_rate,
                'active_executions': len(self.active_executions)
            }

class ResourceManager:
    """Manage execution resources and limits"""
    
    def __init__(self, max_parallel_executions: int = 10, max_memory_mb: int = 1024):
        self.max_parallel_executions = max_parallel_executions
        self.max_memory_mb = max_memory_mb
        self.current_executions = 0
        self.current_memory_usage = 0
        self._execution_semaphore = asyncio.Semaphore(max_parallel_executions)
        self._memory_lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire_execution_slot(self, estimated_memory: int = 0):
        """Acquire an execution slot with memory check"""
        async with self._execution_semaphore:
            async with self._memory_lock:
                if self.current_memory_usage + estimated_memory > self.max_memory_mb * 1024 * 1024:
                    raise RuntimeError(f"Insufficient memory: need {estimated_memory}, "
                                     f"available {self.max_memory_mb * 1024 * 1024 - self.current_memory_usage}")
                
                self.current_memory_usage += estimated_memory
                self.current_executions += 1
            
            try:
                yield
            finally:
                async with self._memory_lock:
                    self.current_memory_usage = max(0, self.current_memory_usage - estimated_memory)
                    self.current_executions = max(0, self.current_executions - 1)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        return {
            'current_executions': self.current_executions,
            'max_parallel_executions': self.max_parallel_executions,
            'current_memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'max_memory_mb': self.max_memory_mb,
            'execution_slots_available': self.max_parallel_executions - self.current_executions,
            'memory_available_mb': self.max_memory_mb - (self.current_memory_usage / (1024 * 1024))
        }

class QueryExecutionEngine:
    """
    High-performance query execution engine for Nova memory system
    Supports parallel execution, monitoring, and adaptive optimization
    """
    
    def __init__(self, optimizer: MemoryQueryOptimizer, 
                 max_workers: int = 4, execution_timeout: float = 300.0):
        self.optimizer = optimizer
        self.max_workers = max_workers
        self.execution_timeout = execution_timeout
        
        # Core components
        self.monitor = ExecutionMonitor()
        self.resource_manager = ResourceManager()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Operation handlers
        self.operation_handlers = {
            'access_layers': self._execute_layer_access,
            'apply_filters': self._execute_filters,
            'full_text_search': self._execute_full_text_search,
            'validate_data': self._execute_validation,
            'insert_data': self._execute_insert,
            'scan_all': self._execute_scan,
            'return_results': self._execute_return,
            'rank_results': self._execute_ranking,
            'aggregate': self._execute_aggregation,
            'join': self._execute_join,
            'sort': self._execute_sort
        }
        
        # Execution cache for intermediate results
        self.intermediate_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info(f"Query Execution Engine initialized with {max_workers} workers")
    
    async def execute_query(self, plan: QueryPlan, context: ExecutionContext) -> ExecutionResult:
        """
        Execute optimized query plan
        Main entry point for query execution
        """
        execution_id = context.execution_id
        start_time = datetime.utcnow()
        
        logger.info(f"Starting execution {execution_id} for plan {plan.plan_id}")
        
        # Start monitoring
        self.monitor.start_execution(execution_id, plan, context)
        
        # Initialize result
        result = ExecutionResult(
            execution_id=execution_id,
            status=ExecutionStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            # Acquire execution resources
            estimated_memory = self._estimate_memory_usage(plan)
            
            async with self.resource_manager.acquire_execution_slot(estimated_memory):
                # Execute the plan
                if plan.parallelizable and len(plan.optimized_operations) > 1:
                    execution_data = await self._execute_parallel(plan, context, result)
                else:
                    execution_data = await self._execute_sequential(plan, context, result)
                
                result.data = execution_data
                result.status = ExecutionStatus.COMPLETED
                
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.CANCELLED
            result.error = "Execution timeout"
            logger.warning(f"Execution {execution_id} timed out")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            logger.error(f"Execution {execution_id} failed: {e}")
            if context.trace_execution:
                result.execution_trace.append({
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        finally:
            # Complete execution
            result.completed_at = datetime.utcnow()
            
            # Create execution statistics
            result.execution_stats = self._create_execution_statistics(
                plan, result, context
            )
            
            # Complete monitoring
            self.monitor.complete_execution(execution_id, result)
            
            # Record stats for optimization learning
            if result.execution_stats:
                await self.optimizer.record_execution_stats(
                    plan.plan_id, result.execution_stats
                )
            
            logger.info(f"Completed execution {execution_id} in "
                       f"{result.execution_time:.3f}s with status {result.status.value}")
        
        return result
    
    async def _execute_parallel(self, plan: QueryPlan, context: ExecutionContext, 
                              result: ExecutionResult) -> Any:
        """Execute operations in parallel"""
        if context.trace_execution:
            result.execution_trace.append({
                'phase': 'parallel_execution_start',
                'operations_count': len(plan.optimized_operations),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Group operations by dependencies
        operation_groups = self._analyze_operation_dependencies(plan.optimized_operations)
        
        execution_data = None
        intermediate_results = {}
        
        # Execute operation groups sequentially, operations within groups in parallel
        for group_id, operations in enumerate(operation_groups):
            if context.trace_execution:
                result.execution_trace.append({
                    'phase': f'executing_group_{group_id}',
                    'operations': [op['operation'] for op in operations],
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Execute operations in this group in parallel
            tasks = []
            for op_id, operation in enumerate(operations):
                task = asyncio.create_task(
                    self._execute_operation(
                        operation, intermediate_results, context, 
                        f"group_{group_id}_op_{op_id}"
                    )
                )
                tasks.append((f"group_{group_id}_op_{op_id}", task))
            
            # Wait for all operations in group to complete
            group_results = {}
            for op_key, task in tasks:
                try:
                    timeout = context.timeout_seconds or self.execution_timeout
                    op_result = await asyncio.wait_for(task, timeout=timeout)
                    group_results[op_key] = op_result
                    
                    # Update intermediate results
                    if op_result.success and op_result.data is not None:
                        intermediate_results[op_key] = op_result.data
                        execution_data = op_result.data  # Use last successful result
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Operation {op_key} timed out")
                    raise
                except Exception as e:
                    logger.error(f"Operation {op_key} failed: {e}")
                    if any(op['operation'] == 'return_results' for op in operations):
                        # Critical operation failed
                        raise
        
        return execution_data
    
    async def _execute_sequential(self, plan: QueryPlan, context: ExecutionContext,
                                result: ExecutionResult) -> Any:
        """Execute operations sequentially"""
        if context.trace_execution:
            result.execution_trace.append({
                'phase': 'sequential_execution_start',
                'operations_count': len(plan.optimized_operations),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        execution_data = None
        intermediate_results = {}
        
        for op_id, operation in enumerate(plan.optimized_operations):
            if context.trace_execution:
                result.execution_trace.append({
                    'phase': f'executing_operation_{op_id}',
                    'operation': operation['operation'],
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Execute operation
            op_result = await self._execute_operation(
                operation, intermediate_results, context, f"seq_op_{op_id}"
            )
            
            if not op_result.success:
                if operation.get('critical', True):
                    raise RuntimeError(f"Critical operation failed: {op_result.error}")
                else:
                    logger.warning(f"Non-critical operation failed: {op_result.error}")
                    continue
            
            # Update results
            if op_result.data is not None:
                intermediate_results[f"seq_op_{op_id}"] = op_result.data
                execution_data = op_result.data
        
        return execution_data
    
    async def _execute_operation(self, operation: Dict[str, Any], 
                               intermediate_results: Dict[str, Any],
                               context: ExecutionContext,
                               operation_id: str) -> OperationResult:
        """Execute a single operation"""
        operation_type = operation['operation']
        start_time = time.time()
        
        try:
            # Get operation handler
            handler = self.operation_handlers.get(operation_type)
            if not handler:
                raise ValueError(f"Unknown operation type: {operation_type}")
            
            # Execute operation
            result_data = await handler(operation, intermediate_results, context)
            
            execution_time = time.time() - start_time
            
            return OperationResult(
                operation_id=operation_id,
                operation_type=operation_type,
                success=True,
                data=result_data,
                execution_time=execution_time,
                rows_processed=self._estimate_rows_processed(result_data),
                memory_used=self._estimate_memory_used(result_data)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Operation {operation_type} failed: {e}")
            
            return OperationResult(
                operation_id=operation_id,
                operation_type=operation_type,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _analyze_operation_dependencies(self, operations: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Analyze operation dependencies for parallel execution"""
        # Simple dependency analysis - group by data flow
        groups = []
        current_group = []
        
        for operation in operations:
            op_type = operation['operation']
            
            # Operations that need previous results
            if op_type in ['apply_filters', 'rank_results', 'return_results'] and current_group:
                groups.append(current_group)
                current_group = [operation]
            else:
                current_group.append(operation)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _execute_layer_access(self, operation: Dict[str, Any], 
                                  intermediate_results: Dict[str, Any],
                                  context: ExecutionContext) -> Any:
        """Execute memory layer access operation"""
        layers = operation.get('layers', [])
        
        # Simulate layer access (in real implementation, this would use the memory router)
        layer_data = {}
        for layer in layers:
            # Simulate data retrieval from layer
            layer_data[f'layer_{layer}'] = {
                'entries': [],  # Would contain actual memory entries
                'metadata': {'layer_id': layer, 'access_time': datetime.utcnow().isoformat()}
            }
        
        return layer_data
    
    async def _execute_filters(self, operation: Dict[str, Any],
                             intermediate_results: Dict[str, Any],
                             context: ExecutionContext) -> Any:
        """Execute filter operation"""
        selectivity = operation.get('selectivity', 1.0)
        
        # Get input data from previous operations
        input_data = None
        for result in intermediate_results.values():
            if isinstance(result, dict) and 'entries' in str(result):
                input_data = result
                break
        
        if input_data is None:
            input_data = {'entries': []}
        
        # Apply filters (simulate)
        filtered_data = input_data.copy()
        if 'entries' in str(filtered_data):
            # Simulate filtering by reducing results based on selectivity
            original_count = len(str(filtered_data))
            filtered_count = int(original_count * selectivity)
            filtered_data['filtered'] = True
            filtered_data['original_count'] = original_count
            filtered_data['filtered_count'] = filtered_count
        
        return filtered_data
    
    async def _execute_full_text_search(self, operation: Dict[str, Any],
                                      intermediate_results: Dict[str, Any],
                                      context: ExecutionContext) -> Any:
        """Execute full-text search operation"""
        use_indexes = operation.get('use_indexes', False)
        
        # Simulate full-text search
        search_results = {
            'matches': [],  # Would contain actual search matches
            'total_matches': 0,
            'search_time': time.time(),
            'used_indexes': use_indexes
        }
        
        return search_results
    
    async def _execute_validation(self, operation: Dict[str, Any],
                                intermediate_results: Dict[str, Any],
                                context: ExecutionContext) -> Any:
        """Execute data validation operation"""
        # Simulate validation
        validation_result = {
            'valid': True,
            'validation_time': time.time(),
            'checks_performed': ['schema', 'constraints', 'permissions']
        }
        
        return validation_result
    
    async def _execute_insert(self, operation: Dict[str, Any],
                            intermediate_results: Dict[str, Any], 
                            context: ExecutionContext) -> Any:
        """Execute data insertion operation"""
        parallel = operation.get('parallel', False)
        
        # Simulate insertion
        insert_result = {
            'inserted_count': 1,
            'insert_time': time.time(),
            'parallel_execution': parallel
        }
        
        return insert_result
    
    async def _execute_scan(self, operation: Dict[str, Any],
                          intermediate_results: Dict[str, Any],
                          context: ExecutionContext) -> Any:
        """Execute scan operation"""
        # Simulate full scan
        scan_result = {
            'scanned_entries': [],  # Would contain scanned data
            'scan_time': time.time(),
            'rows_scanned': 1000  # Simulate
        }
        
        return scan_result
    
    async def _execute_return(self, operation: Dict[str, Any],
                            intermediate_results: Dict[str, Any],
                            context: ExecutionContext) -> Any:
        """Execute return results operation"""
        parallel = operation.get('parallel', True)
        
        # Combine all intermediate results
        combined_results = {
            'results': intermediate_results,
            'parallel_processed': parallel,
            'return_time': time.time()
        }
        
        return combined_results
    
    async def _execute_ranking(self, operation: Dict[str, Any],
                             intermediate_results: Dict[str, Any],
                             context: ExecutionContext) -> Any:
        """Execute result ranking operation"""
        # Simulate ranking
        ranking_result = {
            'ranked_results': [],  # Would contain ranked results
            'ranking_algorithm': 'relevance',
            'ranking_time': time.time()
        }
        
        return ranking_result
    
    async def _execute_aggregation(self, operation: Dict[str, Any],
                                 intermediate_results: Dict[str, Any],
                                 context: ExecutionContext) -> Any:
        """Execute aggregation operation"""
        # Simulate aggregation
        aggregation_result = {
            'aggregated_data': {},
            'aggregation_functions': ['count', 'sum', 'avg'],
            'aggregation_time': time.time()
        }
        
        return aggregation_result
    
    async def _execute_join(self, operation: Dict[str, Any],
                          intermediate_results: Dict[str, Any],
                          context: ExecutionContext) -> Any:
        """Execute join operation"""
        join_type = operation.get('join_type', 'inner')
        
        # Simulate join
        join_result = {
            'joined_data': [],
            'join_type': join_type,
            'join_time': time.time(),
            'rows_joined': 100  # Simulate
        }
        
        return join_result
    
    async def _execute_sort(self, operation: Dict[str, Any],
                          intermediate_results: Dict[str, Any],
                          context: ExecutionContext) -> Any:
        """Execute sort operation"""
        sort_columns = operation.get('columns', [])
        
        # Simulate sorting
        sort_result = {
            'sorted_data': [],
            'sort_columns': sort_columns,
            'sort_time': time.time(),
            'rows_sorted': 100  # Simulate
        }
        
        return sort_result
    
    def _estimate_memory_usage(self, plan: QueryPlan) -> int:
        """Estimate memory usage for plan execution"""
        base_memory = 1024 * 1024  # 1MB base
        
        # Add memory per operation
        operation_memory = len(plan.optimized_operations) * 512 * 1024  # 512KB per operation
        
        # Add memory per layer
        layer_memory = len(plan.memory_layers) * 256 * 1024  # 256KB per layer
        
        return base_memory + operation_memory + layer_memory
    
    def _estimate_rows_processed(self, data: Any) -> int:
        """Estimate number of rows processed"""
        if isinstance(data, dict):
            if 'rows_scanned' in data:
                return data['rows_scanned']
            elif 'rows_joined' in data:
                return data['rows_joined']
            elif 'rows_sorted' in data:
                return data['rows_sorted']
            elif 'entries' in str(data):
                return 100  # Default estimate
        
        return 1  # Minimum
    
    def _estimate_memory_used(self, data: Any) -> int:
        """Estimate memory used by operation"""
        if data is None:
            return 1024  # 1KB minimum
        
        # Simple estimate based on data size
        try:
            data_str = str(data)
            return len(data_str.encode('utf-8'))
        except:
            return 1024  # Default
    
    def _create_execution_statistics(self, plan: QueryPlan, result: ExecutionResult,
                                   context: ExecutionContext) -> ExecutionStatistics:
        """Create execution statistics from result"""
        actual_cost = 0.0
        actual_time = result.execution_time or 0.0
        rows_processed = 0
        memory_usage = 0
        cache_hits = 0
        cache_misses = 0
        errors = []
        
        if result.status == ExecutionStatus.FAILED:
            actual_cost = 1000.0  # High cost for failed execution
            if result.error:
                errors.append(result.error)
        else:
            # Estimate actual cost based on execution time
            actual_cost = max(actual_time * 10, 1.0)
            
            # Extract metrics from execution data
            if isinstance(result.data, dict):
                if 'results' in result.data:
                    # Count metrics from all operations
                    for op_result in result.data['results'].values():
                        if isinstance(op_result, dict):
                            rows_processed += op_result.get('rows_scanned', 0)
                            rows_processed += op_result.get('rows_joined', 0) 
                            rows_processed += op_result.get('rows_sorted', 0)
        
        memory_usage = self._estimate_memory_usage(plan)
        
        return ExecutionStatistics(
            plan_id=plan.plan_id,
            actual_cost=actual_cost,
            actual_time=actual_time,
            rows_processed=rows_processed,
            memory_usage=memory_usage,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            errors=errors,
            execution_timestamp=result.completed_at or datetime.utcnow()
        )
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        # Implementation would cancel the actual execution
        logger.info(f"Cancelling execution {execution_id}")
        return True
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution"""
        with self.monitor._lock:
            if execution_id in self.monitor.active_executions:
                info = self.monitor.active_executions[execution_id]
                return {
                    'execution_id': execution_id,
                    'status': info['status'],
                    'plan_id': info['plan'].plan_id,
                    'started_at': info['started_at'],
                    'duration': (datetime.utcnow() - info['started_at']).total_seconds()
                }
        
        # Check history
        for entry in reversed(self.monitor.execution_history):
            if entry['execution_id'] == execution_id:
                return {
                    'execution_id': execution_id,
                    'status': entry['result'].status,
                    'completed_at': entry['completed_at'],
                    'execution_time': entry['result'].execution_time
                }
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        monitor_metrics = self.monitor.get_performance_metrics()
        resource_status = self.resource_manager.get_resource_status()
        
        return {
            'execution_metrics': monitor_metrics,
            'resource_status': resource_status,
            'engine_config': {
                'max_workers': self.max_workers,
                'execution_timeout': self.execution_timeout,
                'cache_entries': len(self.intermediate_cache)
            }
        }
    
    async def cleanup_cache(self, max_age_seconds: int = None):
        """Clean up expired cache entries"""
        if max_age_seconds is None:
            max_age_seconds = self.cache_ttl
        
        cutoff_time = time.time() - max_age_seconds
        expired_keys = []
        
        for key, (data, timestamp) in self.intermediate_cache.items():
            if timestamp < cutoff_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.intermediate_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def shutdown(self):
        """Shutdown the execution engine"""
        logger.info("Shutting down Query Execution Engine...")
        
        # Cancel all active executions
        active_executions = list(self.monitor.active_executions.keys())
        for execution_id in active_executions:
            await self.cancel_execution(execution_id)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear cache
        self.intermediate_cache.clear()
        
        logger.info("Query Execution Engine shutdown complete")