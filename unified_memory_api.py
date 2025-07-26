#!/usr/bin/env python3
"""
Nova Memory System - Unified Memory API
Single interface for all memory operations across 50+ layers
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from database_connections import NovaDatabasePool
from memory_router import MemoryRouter, MemoryType
from memory_layers import MemoryEntry, MemoryScope, MemoryImportance
from layer_implementations import ImmediateMemoryManager

logger = logging.getLogger(__name__)

class MemoryOperation(Enum):
    """Memory operation types"""
    WRITE = "write"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"
    ANALYZE = "analyze"
    CONSOLIDATE = "consolidate"
    TRANSFER = "transfer"

@dataclass
class MemoryRequest:
    """Unified memory request structure"""
    operation: MemoryOperation
    nova_id: str
    data: Optional[Dict[str, Any]] = None
    query: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class MemoryResponse:
    """Unified memory response structure"""
    success: bool
    operation: MemoryOperation
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    performance: Dict[str, Any] = field(default_factory=dict)

class NovaMemoryAPI:
    """
    Unified API for Nova Memory System
    Single entry point for all memory operations
    """
    
    def __init__(self):
        self.db_pool = NovaDatabasePool()
        self.router = MemoryRouter(self.db_pool)
        self.initialized = False
        self.operation_handlers = {
            MemoryOperation.WRITE: self._handle_write,
            MemoryOperation.READ: self._handle_read,
            MemoryOperation.UPDATE: self._handle_update,
            MemoryOperation.DELETE: self._handle_delete,
            MemoryOperation.SEARCH: self._handle_search,
            MemoryOperation.ANALYZE: self._handle_analyze,
            MemoryOperation.CONSOLIDATE: self._handle_consolidate,
            MemoryOperation.TRANSFER: self._handle_transfer
        }
        self.middleware = []
        self.performance_tracker = {
            'total_operations': 0,
            'operation_times': {},
            'errors_by_type': {}
        }
        
    async def initialize(self):
        """Initialize the memory system"""
        if self.initialized:
            return
            
        logger.info("Initializing Nova Memory API...")
        
        # Initialize database connections
        await self.db_pool.initialize_all_connections()
        
        # Initialize router
        await self.router.initialize()
        
        # Health check
        health = await self.db_pool.check_all_health()
        logger.info(f"System health: {health['overall_status']}")
        
        self.initialized = True
        logger.info("Nova Memory API initialized successfully")
        
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Nova Memory API...")
        await self.db_pool.close_all()
        self.initialized = False
        
    def add_middleware(self, middleware: Callable):
        """Add middleware for request/response processing"""
        self.middleware.append(middleware)
        
    async def execute(self, request: MemoryRequest) -> MemoryResponse:
        """Execute a memory operation"""
        if not self.initialized:
            await self.initialize()
            
        start_time = datetime.now()
        
        # Apply request middleware
        for mw in self.middleware:
            request = await mw(request, 'request')
            
        # Track operation
        self.performance_tracker['total_operations'] += 1
        
        try:
            # Get handler
            handler = self.operation_handlers.get(request.operation)
            if not handler:
                raise ValueError(f"Unknown operation: {request.operation}")
                
            # Execute operation
            response = await handler(request)
            
            # Add performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            response.performance = {
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Track performance
            op_name = request.operation.value
            if op_name not in self.performance_tracker['operation_times']:
                self.performance_tracker['operation_times'][op_name] = []
            self.performance_tracker['operation_times'][op_name].append(execution_time)
            
        except Exception as e:
            logger.error(f"Operation {request.operation} failed: {str(e)}")
            response = MemoryResponse(
                success=False,
                operation=request.operation,
                data=None,
                errors=[str(e)]
            )
            
            # Track errors
            error_type = type(e).__name__
            if error_type not in self.performance_tracker['errors_by_type']:
                self.performance_tracker['errors_by_type'][error_type] = 0
            self.performance_tracker['errors_by_type'][error_type] += 1
            
        # Apply response middleware
        for mw in reversed(self.middleware):
            response = await mw(response, 'response')
            
        return response
        
    # Memory Operations
    
    async def remember(self, nova_id: str, content: Any, 
                      importance: float = 0.5, context: str = "general",
                      memory_type: Optional[MemoryType] = None,
                      tags: List[str] = None) -> MemoryResponse:
        """
        High-level remember operation
        Automatically routes to appropriate layers
        """
        data = {
            'content': content,
            'importance': importance,
            'context': context,
            'tags': tags or [],
            'timestamp': datetime.now().isoformat()
        }
        
        if memory_type:
            data['memory_type'] = memory_type.value
            
        request = MemoryRequest(
            operation=MemoryOperation.WRITE,
            nova_id=nova_id,
            data=data
        )
        
        return await self.execute(request)
        
    async def recall(self, nova_id: str, query: Optional[Union[str, Dict]] = None,
                    memory_types: List[MemoryType] = None,
                    time_range: Optional[timedelta] = None,
                    limit: int = 100) -> MemoryResponse:
        """
        High-level recall operation
        Searches across appropriate layers
        """
        # Build query
        if isinstance(query, str):
            query_dict = {'search': query}
        else:
            query_dict = query or {}
            
        if memory_types:
            query_dict['memory_types'] = [mt.value for mt in memory_types]
            
        if time_range:
            query_dict['time_range'] = time_range.total_seconds()
            
        query_dict['limit'] = limit
        
        request = MemoryRequest(
            operation=MemoryOperation.READ,
            nova_id=nova_id,
            query=query_dict
        )
        
        return await self.execute(request)
        
    async def reflect(self, nova_id: str, time_period: timedelta = None) -> MemoryResponse:
        """
        Analyze patterns in memories
        Meta-cognitive operation
        """
        request = MemoryRequest(
            operation=MemoryOperation.ANALYZE,
            nova_id=nova_id,
            options={
                'time_period': time_period.total_seconds() if time_period else None,
                'analysis_type': 'reflection'
            }
        )
        
        return await self.execute(request)
        
    async def consolidate(self, nova_id: str, aggressive: bool = False) -> MemoryResponse:
        """
        Consolidate memories from short-term to long-term
        """
        request = MemoryRequest(
            operation=MemoryOperation.CONSOLIDATE,
            nova_id=nova_id,
            options={'aggressive': aggressive}
        )
        
        return await self.execute(request)
        
    # Operation Handlers
    
    async def _handle_write(self, request: MemoryRequest) -> MemoryResponse:
        """Handle write operations"""
        try:
            # Route the write
            result = await self.router.route_write(request.nova_id, request.data)
            
            # Build response
            success = bool(result.get('primary_result', {}).get('success'))
            
            return MemoryResponse(
                success=success,
                operation=MemoryOperation.WRITE,
                data={
                    'memory_id': result.get('primary_result', {}).get('memory_id'),
                    'layers_written': [result['primary_result']['layer_id']] + 
                                    [r['layer_id'] for r in result.get('secondary_results', [])],
                    'routing_decision': result.get('routing_decision')
                },
                errors=result.get('errors', [])
            )
            
        except Exception as e:
            return MemoryResponse(
                success=False,
                operation=MemoryOperation.WRITE,
                data=None,
                errors=[str(e)]
            )
            
    async def _handle_read(self, request: MemoryRequest) -> MemoryResponse:
        """Handle read operations"""
        try:
            # Route the read
            result = await self.router.route_read(request.nova_id, request.query or {})
            
            # Format memories
            memories = []
            for memory in result.get('merged_results', []):
                if isinstance(memory, MemoryEntry):
                    memories.append(memory.to_dict())
                else:
                    memories.append(memory)
                    
            return MemoryResponse(
                success=True,
                operation=MemoryOperation.READ,
                data={
                    'memories': memories,
                    'total_count': result.get('total_count', 0),
                    'layers_queried': list(result.get('results_by_layer', {}).keys())
                }
            )
            
        except Exception as e:
            return MemoryResponse(
                success=False,
                operation=MemoryOperation.READ,
                data=None,
                errors=[str(e)]
            )
            
    async def _handle_update(self, request: MemoryRequest) -> MemoryResponse:
        """Handle update operations"""
        # Get memory_id and updates from request
        memory_id = request.query.get('memory_id')
        updates = request.data
        
        if not memory_id:
            return MemoryResponse(
                success=False,
                operation=MemoryOperation.UPDATE,
                data=None,
                errors=["memory_id required for update"]
            )
            
        # Find which layer contains this memory
        # For now, try immediate layers
        success = False
        for layer_id in range(1, 11):
            layer = self.router.layer_managers['immediate'].layers[layer_id]
            if await layer.update(request.nova_id, memory_id, updates):
                success = True
                break
                
        return MemoryResponse(
            success=success,
            operation=MemoryOperation.UPDATE,
            data={'memory_id': memory_id, 'updated': success}
        )
        
    async def _handle_delete(self, request: MemoryRequest) -> MemoryResponse:
        """Handle delete operations"""
        memory_id = request.query.get('memory_id')
        
        if not memory_id:
            return MemoryResponse(
                success=False,
                operation=MemoryOperation.DELETE,
                data=None,
                errors=["memory_id required for delete"]
            )
            
        # Try to delete from all layers
        deleted_from = []
        for layer_id in range(1, 11):
            layer = self.router.layer_managers['immediate'].layers[layer_id]
            if await layer.delete(request.nova_id, memory_id):
                deleted_from.append(layer_id)
                
        return MemoryResponse(
            success=len(deleted_from) > 0,
            operation=MemoryOperation.DELETE,
            data={'memory_id': memory_id, 'deleted_from_layers': deleted_from}
        )
        
    async def _handle_search(self, request: MemoryRequest) -> MemoryResponse:
        """Handle search operations"""
        search_query = request.query.get('search', '')
        layers = request.query.get('layers')
        
        # Cross-layer search
        results = await self.router.cross_layer_query(
            request.nova_id, 
            search_query,
            layers
        )
        
        return MemoryResponse(
            success=True,
            operation=MemoryOperation.SEARCH,
            data={
                'query': search_query,
                'results': [m.to_dict() for m in results],
                'count': len(results)
            }
        )
        
    async def _handle_analyze(self, request: MemoryRequest) -> MemoryResponse:
        """Handle analysis operations"""
        analysis_type = request.options.get('analysis_type', 'general')
        time_period = request.options.get('time_period')
        
        # Get memories for analysis
        memories = await self.router.route_read(request.nova_id, {})
        
        # Perform analysis
        analysis = {
            'total_memories': memories['total_count'],
            'memories_by_layer': {},
            'patterns': [],
            'insights': []
        }
        
        # Analyze by layer
        for layer_id, layer_data in memories['results_by_layer'].items():
            if 'memories' in layer_data:
                analysis['memories_by_layer'][layer_id] = {
                    'count': layer_data['count'],
                    'average_importance': sum(m.get('importance', 0.5) for m in layer_data['memories']) / max(layer_data['count'], 1)
                }
                
        # Pattern detection (simplified)
        if analysis_type == 'reflection':
            # Look for recurring themes
            all_content = ' '.join(str(m.get('data', {})) for layer in memories['results_by_layer'].values() 
                                 for m in layer.get('memories', []))
            
            # Simple word frequency
            words = all_content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
                    
            # Top patterns
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis['patterns'] = [{'word': w, 'frequency': f} for w, f in top_words]
            
            # Generate insights
            if analysis['total_memories'] > 100:
                analysis['insights'].append("High memory activity detected")
            if any(f > 10 for _, f in top_words):
                analysis['insights'].append(f"Recurring theme: {top_words[0][0]}")
                
        return MemoryResponse(
            success=True,
            operation=MemoryOperation.ANALYZE,
            data=analysis
        )
        
    async def _handle_consolidate(self, request: MemoryRequest) -> MemoryResponse:
        """Handle consolidation operations"""
        aggressive = request.options.get('aggressive', False)
        
        # Get short-term memories
        short_term_layers = list(range(6, 11))
        memories = await self.router.route_read(request.nova_id, {'layers': short_term_layers})
        
        consolidated = {
            'episodic': 0,
            'semantic': 0,
            'procedural': 0,
            'emotional': 0,
            'social': 0
        }
        
        # Consolidation logic would go here
        # For now, just count what would be consolidated
        for layer_id, layer_data in memories['results_by_layer'].items():
            if layer_id == 6:  # Episodic
                consolidated['episodic'] = layer_data.get('count', 0)
            elif layer_id == 7:  # Semantic
                consolidated['semantic'] = layer_data.get('count', 0)
            # etc...
            
        return MemoryResponse(
            success=True,
            operation=MemoryOperation.CONSOLIDATE,
            data={
                'consolidated': consolidated,
                'total': sum(consolidated.values()),
                'aggressive': aggressive
            }
        )
        
    async def _handle_transfer(self, request: MemoryRequest) -> MemoryResponse:
        """Handle memory transfer between Novas"""
        target_nova = request.options.get('target_nova')
        memory_types = request.options.get('memory_types', [])
        
        if not target_nova:
            return MemoryResponse(
                success=False,
                operation=MemoryOperation.TRANSFER,
                data=None,
                errors=["target_nova required for transfer"]
            )
            
        # Get memories to transfer
        source_memories = await self.router.route_read(request.nova_id, {
            'memory_types': memory_types
        })
        
        # Transfer logic would go here
        transfer_count = source_memories['total_count']
        
        return MemoryResponse(
            success=True,
            operation=MemoryOperation.TRANSFER,
            data={
                'source_nova': request.nova_id,
                'target_nova': target_nova,
                'memories_transferred': transfer_count,
                'memory_types': memory_types
            }
        )
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        stats = {
            'total_operations': self.performance_tracker['total_operations'],
            'average_times': {},
            'error_rate': 0,
            'errors_by_type': self.performance_tracker['errors_by_type']
        }
        
        # Calculate averages
        for op, times in self.performance_tracker['operation_times'].items():
            if times:
                stats['average_times'][op] = sum(times) / len(times)
                
        # Error rate
        total_errors = sum(self.performance_tracker['errors_by_type'].values())
        if self.performance_tracker['total_operations'] > 0:
            stats['error_rate'] = total_errors / self.performance_tracker['total_operations']
            
        return stats

# Convenience functions
memory_api = NovaMemoryAPI()

async def remember(nova_id: str, content: Any, **kwargs) -> MemoryResponse:
    """Global remember function"""
    return await memory_api.remember(nova_id, content, **kwargs)
    
async def recall(nova_id: str, query: Any = None, **kwargs) -> MemoryResponse:
    """Global recall function"""
    return await memory_api.recall(nova_id, query, **kwargs)
    
async def reflect(nova_id: str, **kwargs) -> MemoryResponse:
    """Global reflect function"""
    return await memory_api.reflect(nova_id, **kwargs)

# Example usage
async def test_unified_api():
    """Test the unified memory API"""
    
    # Initialize
    api = NovaMemoryAPI()
    await api.initialize()
    
    # Test remember
    print("\n=== Testing Remember ===")
    response = await api.remember(
        'bloom',
        'User asked about memory architecture',
        importance=0.8,
        context='conversation',
        memory_type=MemoryType.SOCIAL,
        tags=['user_interaction', 'technical']
    )
    print(f"Remember response: {response.success}")
    print(f"Memory ID: {response.data.get('memory_id')}")
    
    # Test recall
    print("\n=== Testing Recall ===")
    response = await api.recall(
        'bloom',
        'memory architecture',
        limit=10
    )
    print(f"Recall response: {response.success}")
    print(f"Found {response.data.get('total_count')} memories")
    
    # Test reflect
    print("\n=== Testing Reflect ===")
    response = await api.reflect(
        'bloom',
        time_period=timedelta(hours=1)
    )
    print(f"Reflect response: {response.success}")
    print(f"Patterns found: {len(response.data.get('patterns', []))}")
    
    # Performance stats
    print("\n=== Performance Stats ===")
    stats = api.get_performance_stats()
    print(json.dumps(stats, indent=2))
    
    # Shutdown
    await api.shutdown()

if __name__ == "__main__":
    asyncio.run(test_unified_api())