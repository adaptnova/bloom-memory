#!/usr/bin/env python3
"""
Nova Memory System - Intelligent Query Optimizer
Cost-based optimization system for memory queries with caching and adaptive optimization
"""

import json
import asyncio
import logging
import time
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, OrderedDict
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Query optimization levels"""
    MINIMAL = 1
    BALANCED = 2
    AGGRESSIVE = 3

class QueryType(Enum):
    """Query operation types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    AGGREGATE = "aggregate"
    JOIN = "join"
    ANALYZE = "analyze"

class IndexType(Enum):
    """Index recommendation types"""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    VECTOR = "vector"
    SPATIAL = "spatial"

@dataclass
class QueryPlan:
    """Optimized query execution plan"""
    plan_id: str
    query_hash: str
    original_query: Dict[str, Any]
    optimized_operations: List[Dict[str, Any]]
    estimated_cost: float
    estimated_time: float
    memory_layers: List[int]
    databases: List[str]
    parallelizable: bool = True
    index_hints: List[str] = field(default_factory=list)
    cache_strategy: str = "lru"
    created_at: datetime = field(default_factory=datetime.utcnow)
    execution_stats: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionStatistics:
    """Query execution performance statistics"""
    plan_id: str
    actual_cost: float
    actual_time: float
    rows_processed: int
    memory_usage: int
    cache_hits: int
    cache_misses: int
    errors: List[str] = field(default_factory=list)
    execution_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class IndexRecommendation:
    """Index recommendation for performance improvement"""
    table_name: str
    column_names: List[str]
    index_type: IndexType
    estimated_benefit: float
    creation_cost: float
    maintenance_cost: float
    usage_frequency: int
    priority: int = 1

@dataclass
class OptimizationContext:
    """Context information for query optimization"""
    nova_id: str
    session_id: Optional[str]
    current_memory_load: float
    available_indexes: Dict[str, List[str]]
    system_resources: Dict[str, Any]
    historical_patterns: Dict[str, Any]
    user_preferences: Dict[str, Any] = field(default_factory=dict)

class CostModel:
    """Cost estimation model for query operations"""
    
    # Base costs for different operations (in milliseconds)
    OPERATION_COSTS = {
        'scan': 1.0,
        'index_lookup': 0.1,
        'hash_join': 2.0,
        'nested_loop_join': 5.0,
        'sort': 3.0,
        'filter': 0.5,
        'aggregate': 1.5,
        'memory_access': 0.01,
        'disk_access': 10.0,
        'network_access': 50.0
    }
    
    # Memory layer access costs
    LAYER_COSTS = {
        1: 0.001,   # sensory_buffer
        2: 0.002,   # attention_filter
        3: 0.003,   # working_memory
        4: 0.004,   # executive_buffer
        5: 0.005,   # context_stack
        6: 0.01,    # short_term_episodic
        7: 0.01,    # short_term_semantic
        8: 0.01,    # short_term_procedural
        9: 0.01,    # short_term_emotional
        10: 0.01,   # short_term_social
        11: 0.05,   # episodic_consolidation
        12: 0.05,   # semantic_integration
        13: 0.05,   # procedural_compilation
        14: 0.05,   # emotional_patterns
        15: 0.05,   # social_dynamics
        16: 0.1,    # long_term_episodic
        17: 0.1,    # long_term_semantic
        18: 0.1,    # long_term_procedural
        19: 0.1,    # long_term_emotional
        20: 0.1,    # long_term_social
    }
    
    # Database access costs
    DATABASE_COSTS = {
        'dragonfly': 0.005,   # In-memory
        'postgresql': 0.02,   # Disk-based
        'couchdb': 0.03       # Document-based
    }
    
    @staticmethod
    def estimate_operation_cost(operation: str, row_count: int, 
                              selectivity: float = 1.0) -> float:
        """Estimate cost for a single operation"""
        base_cost = CostModel.OPERATION_COSTS.get(operation, 1.0)
        
        # Apply row count scaling
        if operation in ['scan', 'sort']:
            cost = base_cost * row_count * np.log(row_count + 1)
        elif operation in ['index_lookup', 'filter']:
            cost = base_cost * row_count * selectivity
        elif operation in ['hash_join', 'nested_loop_join']:
            cost = base_cost * row_count * selectivity * np.log(row_count + 1)
        else:
            cost = base_cost * row_count * selectivity
            
        return max(cost, 0.001)  # Minimum cost
    
    @staticmethod
    def estimate_layer_cost(layer_id: int, row_count: int) -> float:
        """Estimate cost for accessing a memory layer"""
        base_cost = CostModel.LAYER_COSTS.get(layer_id, 0.01)
        return base_cost * row_count
    
    @staticmethod
    def estimate_database_cost(database: str, row_count: int) -> float:
        """Estimate cost for database access"""
        base_cost = CostModel.DATABASE_COSTS.get(database, 0.02)
        return base_cost * row_count

class QueryPlanCache:
    """LRU cache for query execution plans with adaptive strategies"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.hit_counts = defaultdict(int)
        self.miss_count = 0
        self.total_accesses = 0
        self._lock = threading.RLock()
    
    def _generate_cache_key(self, query: Dict[str, Any], context: OptimizationContext) -> str:
        """Generate cache key from query and context"""
        key_data = {
            'query': query,
            'nova_id': context.nova_id,
            'memory_load': round(context.current_memory_load, 2),
            'available_indexes': sorted(context.available_indexes.keys())
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: Dict[str, Any], context: OptimizationContext) -> Optional[QueryPlan]:
        """Get cached query plan"""
        with self._lock:
            cache_key = self._generate_cache_key(query, context)
            self.total_accesses += 1
            
            if cache_key in self.cache:
                # Check TTL
                if self.access_times[cache_key] > datetime.utcnow() - timedelta(seconds=self.ttl_seconds):
                    # Move to end (most recently used)
                    plan = self.cache[cache_key]
                    del self.cache[cache_key]
                    self.cache[cache_key] = plan
                    self.access_times[cache_key] = datetime.utcnow()
                    self.hit_counts[cache_key] += 1
                    return plan
                else:
                    # Expired
                    del self.cache[cache_key]
                    del self.access_times[cache_key]
                    del self.hit_counts[cache_key]
            
            self.miss_count += 1
            return None
    
    def put(self, query: Dict[str, Any], context: OptimizationContext, plan: QueryPlan):
        """Cache query plan"""
        with self._lock:
            cache_key = self._generate_cache_key(query, context)
            
            # Remove least recently used if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                del self.hit_counts[oldest_key]
            
            self.cache[cache_key] = plan
            self.access_times[cache_key] = datetime.utcnow()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            hit_rate = (self.total_accesses - self.miss_count) / max(self.total_accesses, 1)
            return {
                'total_accesses': self.total_accesses,
                'cache_hits': self.total_accesses - self.miss_count,
                'cache_misses': self.miss_count,
                'hit_rate': hit_rate,
                'cache_size': len(self.cache),
                'max_size': self.max_size
            }
    
    def clear(self):
        """Clear all cached plans"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_counts.clear()
            self.miss_count = 0
            self.total_accesses = 0

class MemoryQueryOptimizer:
    """
    Intelligent query optimizer for Nova memory system
    Provides cost-based optimization with adaptive caching and learning
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.cost_model = CostModel()
        self.plan_cache = QueryPlanCache()
        self.execution_history = []
        self.index_recommendations = []
        self.pattern_analyzer = QueryPatternAnalyzer()
        self.adaptive_optimizer = AdaptiveOptimizer()
        
        # Statistics tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_optimization_time': 0.0,
            'plans_generated': 0,
            'performance_improvements': []
        }
        
        logger.info(f"Memory Query Optimizer initialized with level: {optimization_level.name}")
    
    async def optimize_query(self, query: Dict[str, Any], 
                           context: OptimizationContext) -> QueryPlan:
        """
        Main optimization entry point
        Returns optimized query execution plan
        """
        start_time = time.time()
        self.optimization_stats['total_optimizations'] += 1
        
        try:
            # Check cache first
            cached_plan = self.plan_cache.get(query, context)
            if cached_plan:
                self.optimization_stats['cache_hits'] += 1
                logger.debug(f"Using cached plan: {cached_plan.plan_id}")
                return cached_plan
            
            self.optimization_stats['cache_misses'] += 1
            
            # Generate query hash
            query_hash = self._generate_query_hash(query)
            
            # Analyze query pattern
            query_analysis = await self._analyze_query_structure(query, context)
            
            # Generate initial plan
            initial_plan = await self._generate_initial_plan(query, context, query_analysis)
            
            # Apply optimizations based on level
            optimized_plan = await self._apply_optimizations(initial_plan, context)
            
            # Estimate costs
            await self._estimate_plan_costs(optimized_plan, context)
            
            # Generate index recommendations
            recommendations = await self._generate_index_recommendations(
                optimized_plan, context
            )
            optimized_plan.index_hints = [rec.table_name for rec in recommendations]
            
            # Cache the plan
            self.plan_cache.put(query, context, optimized_plan)
            self.optimization_stats['plans_generated'] += 1
            
            # Update statistics
            optimization_time = time.time() - start_time
            self._update_optimization_stats(optimization_time)
            
            logger.info(f"Query optimized in {optimization_time:.3f}s, "
                       f"estimated cost: {optimized_plan.estimated_cost:.2f}")
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            # Return simple fallback plan
            return await self._generate_fallback_plan(query, context)
    
    async def record_execution_stats(self, plan_id: str, stats: ExecutionStatistics):
        """Record actual execution statistics for learning"""
        self.execution_history.append(stats)
        
        # Limit history size
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-5000:]
        
        # Update adaptive optimization
        await self.adaptive_optimizer.learn_from_execution(plan_id, stats)
        
        # Update performance improvement tracking
        await self._update_performance_tracking(plan_id, stats)
    
    async def get_index_recommendations(self, limit: int = 10) -> List[IndexRecommendation]:
        """Get top index recommendations for performance improvement"""
        # Sort by estimated benefit
        sorted_recommendations = sorted(
            self.index_recommendations,
            key=lambda r: r.estimated_benefit,
            reverse=True
        )
        return sorted_recommendations[:limit]
    
    async def analyze_query_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze query patterns for optimization insights"""
        return await self.pattern_analyzer.analyze_patterns(
            self.execution_history, time_window_hours
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        cache_stats = self.plan_cache.get_statistics()
        
        return {
            **self.optimization_stats,
            'cache_statistics': cache_stats,
            'execution_history_size': len(self.execution_history),
            'index_recommendations': len(self.index_recommendations),
            'optimization_level': self.optimization_level.name
        }
    
    def _generate_query_hash(self, query: Dict[str, Any]) -> str:
        """Generate hash for query identification"""
        return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()[:16]
    
    async def _analyze_query_structure(self, query: Dict[str, Any], 
                                     context: OptimizationContext) -> Dict[str, Any]:
        """Analyze query structure and requirements"""
        analysis = {
            'query_type': self._determine_query_type(query),
            'complexity': self._calculate_query_complexity(query),
            'memory_layers_needed': self._identify_memory_layers(query),
            'databases_needed': self._identify_databases(query, context),
            'selectivity': self._estimate_selectivity(query),
            'parallelizable': self._check_parallelizability(query)
        }
        
        return analysis
    
    def _determine_query_type(self, query: Dict[str, Any]) -> QueryType:
        """Determine the primary query type"""
        if 'operation' in query:
            op = query['operation'].lower()
            if op in ['read', 'get', 'find']:
                return QueryType.SELECT
            elif op in ['write', 'insert', 'create']:
                return QueryType.INSERT
            elif op in ['update', 'modify']:
                return QueryType.UPDATE
            elif op in ['delete', 'remove']:
                return QueryType.DELETE
            elif op in ['search', 'query']:
                return QueryType.SEARCH
            elif op in ['analyze', 'aggregate']:
                return QueryType.AGGREGATE
        
        return QueryType.SELECT  # Default
    
    def _calculate_query_complexity(self, query: Dict[str, Any]) -> float:
        """Calculate query complexity score (0-10)"""
        complexity = 1.0
        
        # Check for joins
        if 'joins' in query or 'relationships' in query:
            complexity += 2.0
        
        # Check for aggregations
        if 'aggregations' in query or 'group_by' in query:
            complexity += 1.5
        
        # Check for subqueries
        if 'subqueries' in query or isinstance(query.get('conditions'), dict):
            complexity += 1.0
        
        # Check for sorting
        if 'sort' in query or 'order_by' in query:
            complexity += 0.5
        
        # Check for filters
        if 'filters' in query or 'where' in query:
            complexity += 0.5
        
        return min(complexity, 10.0)
    
    def _identify_memory_layers(self, query: Dict[str, Any]) -> List[int]:
        """Identify which memory layers the query needs to access"""
        layers = []
        
        # Extract memory types from query
        memory_types = query.get('memory_types', [])
        scope = query.get('scope', 'working')
        
        # Map to layers based on routing logic
        if 'sensory' in memory_types or scope == 'immediate':
            layers.extend([1, 2])
        if 'working' in memory_types or scope == 'working':
            layers.extend([3, 4, 5])
        if 'episodic' in memory_types or scope == 'episodic':
            layers.extend([6, 11, 16])
        if 'semantic' in memory_types or scope == 'semantic':
            layers.extend([7, 12, 17])
        if 'procedural' in memory_types or scope == 'procedural':
            layers.extend([8, 13, 18])
        
        # Default to working memory if nothing specified
        if not layers:
            layers = [3, 4, 5]
        
        return sorted(list(set(layers)))
    
    def _identify_databases(self, query: Dict[str, Any], 
                          context: OptimizationContext) -> List[str]:
        """Identify which databases the query needs to access"""
        databases = []
        
        # Check query preferences
        if 'databases' in query:
            return query['databases']
        
        # Infer from memory layers
        layers = self._identify_memory_layers(query)
        
        # Short-term layers use DragonflyDB
        if any(layer <= 10 for layer in layers):
            databases.append('dragonfly')
        
        # Long-term layers use PostgreSQL and CouchDB
        if any(layer > 15 for layer in layers):
            databases.extend(['postgresql', 'couchdb'])
        
        # Default to DragonflyDB
        if not databases:
            databases = ['dragonfly']
        
        return list(set(databases))
    
    def _estimate_selectivity(self, query: Dict[str, Any]) -> float:
        """Estimate query selectivity (fraction of data returned)"""
        # Default selectivity
        selectivity = 1.0
        
        # Check for filters
        conditions = query.get('conditions', {})
        if conditions:
            # Estimate based on condition types
            for condition in conditions.values() if isinstance(conditions, dict) else [conditions]:
                if isinstance(condition, dict):
                    if 'equals' in str(condition):
                        selectivity *= 0.1  # Equality is very selective
                    elif 'range' in str(condition) or 'between' in str(condition):
                        selectivity *= 0.3  # Range is moderately selective
                    elif 'like' in str(condition) or 'contains' in str(condition):
                        selectivity *= 0.5  # Pattern matching is less selective
        
        # Check for limits
        if 'limit' in query:
            limit_selectivity = min(query['limit'] / 1000, 1.0)  # Assume 1000 total rows
            selectivity = min(selectivity, limit_selectivity)
        
        return max(selectivity, 0.001)  # Minimum selectivity
    
    def _check_parallelizability(self, query: Dict[str, Any]) -> bool:
        """Check if query can be parallelized"""
        # Queries with ordering dependencies can't be fully parallelized
        if 'sort' in query or 'order_by' in query:
            return False
        
        # Aggregations with GROUP BY can be parallelized
        if 'group_by' in query:
            return True
        
        # Most read operations can be parallelized
        query_type = self._determine_query_type(query)
        return query_type in [QueryType.SELECT, QueryType.SEARCH, QueryType.ANALYZE]
    
    async def _generate_initial_plan(self, query: Dict[str, Any], 
                                   context: OptimizationContext,
                                   analysis: Dict[str, Any]) -> QueryPlan:
        """Generate initial query execution plan"""
        plan_id = f"plan_{int(time.time() * 1000000)}"
        query_hash = self._generate_query_hash(query)
        
        # Generate operations based on query type
        operations = []
        
        if analysis['query_type'] == QueryType.SELECT:
            operations.extend([
                {'operation': 'access_layers', 'layers': analysis['memory_layers_needed']},
                {'operation': 'apply_filters', 'selectivity': analysis['selectivity']},
                {'operation': 'return_results', 'parallel': analysis['parallelizable']}
            ])
        elif analysis['query_type'] == QueryType.INSERT:
            operations.extend([
                {'operation': 'validate_data', 'parallel': False},
                {'operation': 'access_layers', 'layers': analysis['memory_layers_needed']},
                {'operation': 'insert_data', 'parallel': analysis['parallelizable']}
            ])
        elif analysis['query_type'] == QueryType.SEARCH:
            operations.extend([
                {'operation': 'access_layers', 'layers': analysis['memory_layers_needed']},
                {'operation': 'full_text_search', 'parallel': True},
                {'operation': 'rank_results', 'parallel': False},
                {'operation': 'apply_filters', 'selectivity': analysis['selectivity']},
                {'operation': 'return_results', 'parallel': True}
            ])
        
        return QueryPlan(
            plan_id=plan_id,
            query_hash=query_hash,
            original_query=query,
            optimized_operations=operations,
            estimated_cost=0.0,  # Will be calculated later
            estimated_time=0.0,  # Will be calculated later
            memory_layers=analysis['memory_layers_needed'],
            databases=analysis['databases_needed'],
            parallelizable=analysis['parallelizable']
        )
    
    async def _apply_optimizations(self, plan: QueryPlan, 
                                 context: OptimizationContext) -> QueryPlan:
        """Apply optimization rules based on optimization level"""
        if self.optimization_level == OptimizationLevel.MINIMAL:
            return plan
        
        # Rule-based optimizations
        optimized_operations = []
        
        for op in plan.optimized_operations:
            if op['operation'] == 'access_layers':
                # Optimize layer access order
                op['layers'] = self._optimize_layer_access_order(op['layers'], context)
            elif op['operation'] == 'apply_filters':
                # Push filters down closer to data access
                op['push_down'] = True
            elif op['operation'] == 'full_text_search':
                # Use indexes if available
                op['use_indexes'] = True
            
            optimized_operations.append(op)
        
        # Add parallel execution hints for aggressive optimization
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            for op in optimized_operations:
                if op.get('parallel', True):
                    op['parallel_workers'] = min(4, len(plan.memory_layers))
        
        plan.optimized_operations = optimized_operations
        return plan
    
    def _optimize_layer_access_order(self, layers: List[int], 
                                   context: OptimizationContext) -> List[int]:
        """Optimize the order of memory layer access"""
        # Sort by access cost (lower cost first)
        layer_costs = [(layer, self.cost_model.estimate_layer_cost(layer, 1000)) 
                      for layer in layers]
        layer_costs.sort(key=lambda x: x[1])
        return [layer for layer, _ in layer_costs]
    
    async def _estimate_plan_costs(self, plan: QueryPlan, context: OptimizationContext):
        """Estimate execution costs for the plan"""
        total_cost = 0.0
        total_time = 0.0
        
        estimated_rows = 1000  # Default estimate
        
        for op in plan.optimized_operations:
            operation_type = op['operation']
            
            if operation_type == 'access_layers':
                for layer in op['layers']:
                    total_cost += self.cost_model.estimate_layer_cost(layer, estimated_rows)
                    total_time += total_cost  # Simplified time estimate
            elif operation_type == 'apply_filters':
                selectivity = op.get('selectivity', 1.0)
                total_cost += self.cost_model.estimate_operation_cost('filter', estimated_rows, selectivity)
                estimated_rows = int(estimated_rows * selectivity)
            elif operation_type == 'full_text_search':
                total_cost += self.cost_model.estimate_operation_cost('scan', estimated_rows)
            else:
                total_cost += self.cost_model.estimate_operation_cost('scan', estimated_rows)
        
        # Apply database access costs
        for db in plan.databases:
            total_cost += self.cost_model.estimate_database_cost(db, estimated_rows)
        
        # Apply parallelization benefits
        if plan.parallelizable and len(plan.memory_layers) > 1:
            parallel_factor = min(0.5, 1.0 / len(plan.memory_layers))
            total_time *= (1 - parallel_factor)
        
        plan.estimated_cost = total_cost
        plan.estimated_time = total_time
    
    async def _generate_index_recommendations(self, plan: QueryPlan, 
                                            context: OptimizationContext) -> List[IndexRecommendation]:
        """Generate index recommendations based on query plan"""
        recommendations = []
        
        # Analyze operations for index opportunities
        for op in plan.optimized_operations:
            if op['operation'] == 'apply_filters':
                # Recommend indexes for filter conditions
                for table in ['memory_entries', 'episodic_memories', 'semantic_memories']:
                    rec = IndexRecommendation(
                        table_name=table,
                        column_names=['timestamp', 'nova_id'],
                        index_type=IndexType.BTREE,
                        estimated_benefit=plan.estimated_cost * 0.3,
                        creation_cost=10.0,
                        maintenance_cost=1.0,
                        usage_frequency=1,
                        priority=2
                    )
                    recommendations.append(rec)
            elif op['operation'] == 'full_text_search':
                # Recommend text search indexes
                for table in ['semantic_memories', 'episodic_memories']:
                    rec = IndexRecommendation(
                        table_name=table,
                        column_names=['content', 'summary'],
                        index_type=IndexType.GIN,
                        estimated_benefit=plan.estimated_cost * 0.5,
                        creation_cost=20.0,
                        maintenance_cost=2.0,
                        usage_frequency=1,
                        priority=1
                    )
                    recommendations.append(rec)
        
        # Add to global recommendations
        self.index_recommendations.extend(recommendations)
        
        # Remove duplicates and sort by priority
        unique_recommendations = {}
        for rec in self.index_recommendations:
            key = f"{rec.table_name}:{':'.join(rec.column_names)}"
            if key not in unique_recommendations or rec.priority < unique_recommendations[key].priority:
                unique_recommendations[key] = rec
        
        self.index_recommendations = list(unique_recommendations.values())
        self.index_recommendations.sort(key=lambda x: (x.priority, -x.estimated_benefit))
        
        return recommendations
    
    async def _generate_fallback_plan(self, query: Dict[str, Any], 
                                    context: OptimizationContext) -> QueryPlan:
        """Generate simple fallback plan when optimization fails"""
        plan_id = f"fallback_{int(time.time() * 1000000)}"
        query_hash = self._generate_query_hash(query)
        
        return QueryPlan(
            plan_id=plan_id,
            query_hash=query_hash,
            original_query=query,
            optimized_operations=[
                {'operation': 'access_layers', 'layers': [3]},  # Working memory only
                {'operation': 'scan_all', 'parallel': False},
                {'operation': 'return_results', 'parallel': False}
            ],
            estimated_cost=100.0,  # High cost for fallback
            estimated_time=100.0,
            memory_layers=[3],
            databases=['dragonfly'],
            parallelizable=False
        )
    
    def _update_optimization_stats(self, optimization_time: float):
        """Update optimization statistics"""
        current_avg = self.optimization_stats['avg_optimization_time']
        total_opts = self.optimization_stats['total_optimizations']
        
        # Update running average
        new_avg = ((current_avg * (total_opts - 1)) + optimization_time) / total_opts
        self.optimization_stats['avg_optimization_time'] = new_avg
    
    async def _update_performance_tracking(self, plan_id: str, stats: ExecutionStatistics):
        """Update performance improvement tracking"""
        # Find the plan
        for plan in [item for item in self.plan_cache.cache.values() if item.plan_id == plan_id]:
            if plan.estimated_cost > 0:
                improvement = (plan.estimated_cost - stats.actual_cost) / plan.estimated_cost
                self.optimization_stats['performance_improvements'].append({
                    'plan_id': plan_id,
                    'estimated_cost': plan.estimated_cost,
                    'actual_cost': stats.actual_cost,
                    'improvement': improvement,
                    'timestamp': stats.execution_timestamp
                })
                
                # Keep only recent improvements
                if len(self.optimization_stats['performance_improvements']) > 1000:
                    self.optimization_stats['performance_improvements'] = \
                        self.optimization_stats['performance_improvements'][-500:]
            break

class QueryPatternAnalyzer:
    """Analyzes query patterns for optimization insights"""
    
    async def analyze_patterns(self, execution_history: List[ExecutionStatistics], 
                             time_window_hours: int) -> Dict[str, Any]:
        """Analyze execution patterns"""
        if not execution_history:
            return {'patterns': [], 'recommendations': []}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_history = [
            stat for stat in execution_history 
            if stat.execution_timestamp > cutoff_time
        ]
        
        patterns = {
            'query_frequency': self._analyze_query_frequency(recent_history),
            'performance_trends': self._analyze_performance_trends(recent_history),
            'resource_usage': self._analyze_resource_usage(recent_history),
            'error_patterns': self._analyze_error_patterns(recent_history),
            'temporal_patterns': self._analyze_temporal_patterns(recent_history)
        }
        
        recommendations = self._generate_pattern_recommendations(patterns)
        
        return {
            'patterns': patterns,
            'recommendations': recommendations,
            'analysis_window': time_window_hours,
            'total_queries': len(recent_history)
        }
    
    def _analyze_query_frequency(self, history: List[ExecutionStatistics]) -> Dict[str, Any]:
        """Analyze query frequency patterns"""
        plan_counts = defaultdict(int)
        for stat in history:
            plan_counts[stat.plan_id] += 1
        
        return {
            'most_frequent_plans': sorted(plan_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'total_unique_plans': len(plan_counts),
            'avg_executions_per_plan': np.mean(list(plan_counts.values())) if plan_counts else 0
        }
    
    def _analyze_performance_trends(self, history: List[ExecutionStatistics]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if not history:
            return {}
        
        times = [stat.actual_time for stat in history]
        costs = [stat.actual_cost for stat in history]
        
        return {
            'avg_execution_time': np.mean(times),
            'median_execution_time': np.median(times),
            'max_execution_time': np.max(times),
            'avg_cost': np.mean(costs),
            'performance_variance': np.var(times)
        }
    
    def _analyze_resource_usage(self, history: List[ExecutionStatistics]) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        memory_usage = [stat.memory_usage for stat in history if stat.memory_usage > 0]
        rows_processed = [stat.rows_processed for stat in history if stat.rows_processed > 0]
        
        return {
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'max_memory_usage': np.max(memory_usage) if memory_usage else 0,
            'avg_rows_processed': np.mean(rows_processed) if rows_processed else 0,
            'max_rows_processed': np.max(rows_processed) if rows_processed else 0
        }
    
    def _analyze_error_patterns(self, history: List[ExecutionStatistics]) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_counts = defaultdict(int)
        total_errors = 0
        
        for stat in history:
            if stat.errors:
                total_errors += len(stat.errors)
                for error in stat.errors:
                    error_counts[error] += 1
        
        return {
            'total_errors': total_errors,
            'error_rate': total_errors / len(history) if history else 0,
            'most_common_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _analyze_temporal_patterns(self, history: List[ExecutionStatistics]) -> Dict[str, Any]:
        """Analyze temporal execution patterns"""
        if not history:
            return {}
        
        hourly_counts = defaultdict(int)
        for stat in history:
            hour = stat.execution_timestamp.hour
            hourly_counts[hour] += 1
        
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else 0
        
        return {
            'hourly_distribution': dict(hourly_counts),
            'peak_hour': peak_hour,
            'queries_at_peak': hourly_counts[peak_hour]
        }
    
    def _generate_pattern_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on patterns"""
        recommendations = []
        
        # Performance recommendations
        if patterns.get('performance_trends', {}).get('performance_variance', 0) > 100:
            recommendations.append("High performance variance detected. Consider query plan stabilization.")
        
        # Caching recommendations
        freq_patterns = patterns.get('query_frequency', {})
        if freq_patterns.get('total_unique_plans', 0) < freq_patterns.get('avg_executions_per_plan', 0) * 5:
            recommendations.append("Few unique query plans with high reuse. Increase cache size.")
        
        # Error recommendations
        error_rate = patterns.get('error_patterns', {}).get('error_rate', 0)
        if error_rate > 0.1:
            recommendations.append(f"High error rate ({error_rate:.1%}). Review query validation.")
        
        # Resource recommendations
        resource_usage = patterns.get('resource_usage', {})
        if resource_usage.get('max_memory_usage', 0) > 1000000:  # 1MB threshold
            recommendations.append("High memory usage detected. Consider result streaming.")
        
        return recommendations

class AdaptiveOptimizer:
    """Adaptive optimization engine that learns from execution history"""
    
    def __init__(self):
        self.learning_data = defaultdict(list)
        self.adaptation_rules = {}
    
    async def learn_from_execution(self, plan_id: str, stats: ExecutionStatistics):
        """Learn from query execution results"""
        self.learning_data[plan_id].append(stats)
        
        # Adapt optimization rules based on performance
        await self._update_adaptation_rules(plan_id, stats)
    
    async def _update_adaptation_rules(self, plan_id: str, stats: ExecutionStatistics):
        """Update adaptive optimization rules"""
        plan_stats = self.learning_data[plan_id]
        
        if len(plan_stats) >= 5:  # Need enough data points
            recent_performance = [s.actual_time for s in plan_stats[-5:]]
            avg_performance = np.mean(recent_performance)
            
            # Create adaptation rule if performance is consistently poor
            if avg_performance > 100:  # 100ms threshold
                self.adaptation_rules[plan_id] = {
                    'rule': 'increase_parallelism',
                    'confidence': min(len(plan_stats) / 10, 1.0),
                    'last_updated': datetime.utcnow()
                }
            elif avg_performance < 10:  # Very fast queries
                self.adaptation_rules[plan_id] = {
                    'rule': 'reduce_optimization_overhead',
                    'confidence': min(len(plan_stats) / 10, 1.0),
                    'last_updated': datetime.utcnow()
                }
    
    def get_adaptation_suggestions(self, plan_id: str) -> List[str]:
        """Get adaptation suggestions for a query plan"""
        suggestions = []
        
        if plan_id in self.adaptation_rules:
            rule = self.adaptation_rules[plan_id]
            if rule['confidence'] > 0.7:
                suggestions.append(f"Apply {rule['rule']} (confidence: {rule['confidence']:.2f})")
        
        return suggestions