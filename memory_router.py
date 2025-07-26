#!/usr/bin/env python3
"""
Nova Memory System - Intelligent Memory Router
Routes memory operations to appropriate layers and databases
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from database_connections import NovaDatabasePool
from memory_layers import MemoryEntry, MemoryScope, MemoryImportance
from layer_implementations import ImmediateMemoryManager

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Memory type classifications for routing"""
    SENSORY = "sensory"
    ATTENTION = "attention"
    WORKING = "working"
    TASK = "task"
    CONTEXT = "context"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    METACOGNITIVE = "metacognitive"
    PREDICTIVE = "predictive"
    CREATIVE = "creative"
    LINGUISTIC = "linguistic"
    COLLECTIVE = "collective"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"

@dataclass
class RoutingDecision:
    """Routing decision for memory operation"""
    primary_layer: int
    secondary_layers: List[int]
    databases: List[str]
    priority: float
    parallel: bool = True
    
class MemoryRouter:
    """
    Intelligent router that determines which layers and databases
    should handle different types of memory operations
    """
    
    # Layer routing map based on memory type
    TYPE_TO_LAYERS = {
        MemoryType.SENSORY: {
            'primary': 1,  # sensory_buffer
            'secondary': [2],  # attention_filter
            'databases': ['dragonfly']
        },
        MemoryType.ATTENTION: {
            'primary': 2,  # attention_filter
            'secondary': [3],  # working_memory
            'databases': ['dragonfly']
        },
        MemoryType.WORKING: {
            'primary': 3,  # working_memory
            'secondary': [4, 5],  # executive_buffer, context_stack
            'databases': ['dragonfly']
        },
        MemoryType.TASK: {
            'primary': 4,  # executive_buffer
            'secondary': [3, 28],  # working_memory, planning_memory
            'databases': ['dragonfly', 'postgresql']
        },
        MemoryType.CONTEXT: {
            'primary': 5,  # context_stack
            'secondary': [3],  # working_memory
            'databases': ['dragonfly']
        },
        MemoryType.EPISODIC: {
            'primary': 6,  # short_term_episodic
            'secondary': [11, 16],  # episodic_consolidation, long_term_episodic
            'databases': ['dragonfly', 'postgresql']
        },
        MemoryType.SEMANTIC: {
            'primary': 7,  # short_term_semantic
            'secondary': [12, 17],  # semantic_integration, long_term_semantic
            'databases': ['dragonfly', 'couchdb']
        },
        MemoryType.PROCEDURAL: {
            'primary': 8,  # short_term_procedural
            'secondary': [13, 18],  # procedural_compilation, long_term_procedural
            'databases': ['dragonfly', 'postgresql']
        },
        MemoryType.EMOTIONAL: {
            'primary': 9,  # short_term_emotional
            'secondary': [14, 19],  # emotional_patterns, long_term_emotional
            'databases': ['dragonfly', 'arangodb']
        },
        MemoryType.SOCIAL: {
            'primary': 10,  # short_term_social
            'secondary': [15, 20],  # social_models, long_term_social
            'databases': ['dragonfly', 'arangodb']
        },
        MemoryType.METACOGNITIVE: {
            'primary': 21,  # metacognitive_monitoring
            'secondary': [22, 23, 24, 25],  # strategy, error, success, learning
            'databases': ['clickhouse', 'postgresql']
        },
        MemoryType.PREDICTIVE: {
            'primary': 26,  # predictive_models
            'secondary': [27, 28, 29, 30],  # simulation, planning, intention, expectation
            'databases': ['clickhouse', 'arangodb']
        },
        MemoryType.CREATIVE: {
            'primary': 31,  # creative_combinations
            'secondary': [32, 33, 34, 35],  # imaginative, dream, inspiration, aesthetic
            'databases': ['couchdb', 'arangodb']
        },
        MemoryType.LINGUISTIC: {
            'primary': 36,  # linguistic_patterns
            'secondary': [37, 38, 39, 40],  # dialogue, narrative, metaphor, humor
            'databases': ['meilisearch', 'postgresql', 'couchdb']
        },
        MemoryType.COLLECTIVE: {
            'primary': 41,  # collective_knowledge
            'secondary': [42, 43, 44, 45],  # experience, skills, emotions, goals
            'databases': ['arangodb', 'clickhouse', 'dragonfly']
        },
        MemoryType.SPATIAL: {
            'primary': 46,  # spatial_memory
            'secondary': [],
            'databases': ['postgresql']  # PostGIS extension
        },
        MemoryType.TEMPORAL: {
            'primary': 47,  # temporal_memory
            'secondary': [26],  # predictive_models
            'databases': ['clickhouse']
        }
    }
    
    def __init__(self, database_pool: NovaDatabasePool):
        self.database_pool = database_pool
        self.layer_managers = {
            'immediate': ImmediateMemoryManager()  # Layers 1-10
            # Add more managers as implemented
        }
        self.routing_cache = {}  # Cache routing decisions
        self.performance_metrics = {
            'total_routes': 0,
            'cache_hits': 0,
            'routing_errors': 0
        }
        
    async def initialize(self):
        """Initialize all layer managers"""
        # Initialize immediate layers with DragonflyDB
        dragonfly_conn = self.database_pool.get_connection('dragonfly')
        await self.layer_managers['immediate'].initialize_all(dragonfly_conn)
        
        logger.info("Memory router initialized")
        
    def analyze_memory_content(self, data: Dict[str, Any]) -> Set[MemoryType]:
        """Analyze content to determine memory types"""
        memory_types = set()
        
        # Check for explicit type
        if 'memory_type' in data:
            try:
                memory_types.add(MemoryType(data['memory_type']))
            except ValueError:
                pass
                
        # Content analysis
        content = str(data).lower()
        
        # Sensory indicators
        if any(word in content for word in ['see', 'hear', 'feel', 'sense', 'detect']):
            memory_types.add(MemoryType.SENSORY)
            
        # Task indicators
        if any(word in content for word in ['task', 'goal', 'todo', 'plan', 'objective']):
            memory_types.add(MemoryType.TASK)
            
        # Emotional indicators
        if any(word in content for word in ['feel', 'emotion', 'mood', 'happy', 'sad', 'angry']):
            memory_types.add(MemoryType.EMOTIONAL)
            
        # Social indicators
        if any(word in content for word in ['user', 'person', 'interaction', 'conversation', 'social']):
            memory_types.add(MemoryType.SOCIAL)
            
        # Knowledge indicators
        if any(word in content for word in ['know', 'learn', 'understand', 'concept', 'idea']):
            memory_types.add(MemoryType.SEMANTIC)
            
        # Event indicators
        if any(word in content for word in ['event', 'happened', 'occurred', 'experience']):
            memory_types.add(MemoryType.EPISODIC)
            
        # Skill indicators
        if any(word in content for word in ['how to', 'procedure', 'method', 'skill', 'technique']):
            memory_types.add(MemoryType.PROCEDURAL)
            
        # Creative indicators
        if any(word in content for word in ['imagine', 'create', 'idea', 'novel', 'innovative']):
            memory_types.add(MemoryType.CREATIVE)
            
        # Predictive indicators
        if any(word in content for word in ['predict', 'expect', 'future', 'will', 'anticipate']):
            memory_types.add(MemoryType.PREDICTIVE)
            
        # Default to working memory if no specific type identified
        if not memory_types:
            memory_types.add(MemoryType.WORKING)
            
        return memory_types
        
    def calculate_importance(self, data: Dict[str, Any], memory_types: Set[MemoryType]) -> float:
        """Calculate importance score for routing priority"""
        base_importance = data.get('importance', 0.5)
        
        # Boost importance for certain memory types
        type_boosts = {
            MemoryType.TASK: 0.2,
            MemoryType.EMOTIONAL: 0.15,
            MemoryType.METACOGNITIVE: 0.15,
            MemoryType.COLLECTIVE: 0.1
        }
        
        for memory_type in memory_types:
            base_importance += type_boosts.get(memory_type, 0)
            
        # Cap at 1.0
        return min(base_importance, 1.0)
        
    def get_routing_decision(self, data: Dict[str, Any]) -> RoutingDecision:
        """Determine routing for memory operation"""
        # Check cache
        cache_key = hash(json.dumps(data, sort_keys=True))
        if cache_key in self.routing_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.routing_cache[cache_key]
            
        # Analyze content
        memory_types = self.analyze_memory_content(data)
        importance = self.calculate_importance(data, memory_types)
        
        # Collect all relevant layers and databases
        all_layers = set()
        all_databases = set()
        
        for memory_type in memory_types:
            if memory_type in self.TYPE_TO_LAYERS:
                config = self.TYPE_TO_LAYERS[memory_type]
                all_layers.add(config['primary'])
                all_layers.update(config['secondary'])
                all_databases.update(config['databases'])
                
        # Determine primary layer (lowest number = highest priority)
        primary_layer = min(all_layers) if all_layers else 3  # Default to working memory
        secondary_layers = sorted(all_layers - {primary_layer})
        
        # Create routing decision
        decision = RoutingDecision(
            primary_layer=primary_layer,
            secondary_layers=secondary_layers[:5],  # Limit to 5 secondary layers
            databases=list(all_databases),
            priority=importance,
            parallel=len(secondary_layers) > 2  # Parallel if many layers
        )
        
        # Cache decision
        self.routing_cache[cache_key] = decision
        
        # Update metrics
        self.performance_metrics['total_routes'] += 1
        
        return decision
        
    async def route_write(self, nova_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Route a write operation to appropriate layers"""
        # Get routing decision
        decision = self.get_routing_decision(data)
        
        # Prepare write results
        results = {
            'routing_decision': decision,
            'primary_result': None,
            'secondary_results': [],
            'errors': []
        }
        
        try:
            # Write to primary layer
            if decision.primary_layer <= 10:  # Immediate layers
                manager = self.layer_managers['immediate']
                layer = manager.layers[decision.primary_layer]
                memory_id = await layer.write(nova_id, data, importance=decision.priority)
                results['primary_result'] = {
                    'layer_id': decision.primary_layer,
                    'memory_id': memory_id,
                    'success': True
                }
            
            # Write to secondary layers
            if decision.secondary_layers:
                if decision.parallel:
                    # Parallel writes
                    tasks = []
                    for layer_id in decision.secondary_layers:
                        if layer_id <= 10:
                            layer = self.layer_managers['immediate'].layers[layer_id]
                            tasks.append(layer.write(nova_id, data, importance=decision.priority))
                            
                    if tasks:
                        secondary_ids = await asyncio.gather(*tasks, return_exceptions=True)
                        for i, result in enumerate(secondary_ids):
                            if isinstance(result, Exception):
                                results['errors'].append(str(result))
                            else:
                                results['secondary_results'].append({
                                    'layer_id': decision.secondary_layers[i],
                                    'memory_id': result,
                                    'success': True
                                })
                else:
                    # Sequential writes
                    for layer_id in decision.secondary_layers:
                        if layer_id <= 10:
                            try:
                                layer = self.layer_managers['immediate'].layers[layer_id]
                                memory_id = await layer.write(nova_id, data, importance=decision.priority)
                                results['secondary_results'].append({
                                    'layer_id': layer_id,
                                    'memory_id': memory_id,
                                    'success': True
                                })
                            except Exception as e:
                                results['errors'].append(f"Layer {layer_id}: {str(e)}")
                                
        except Exception as e:
            self.performance_metrics['routing_errors'] += 1
            results['errors'].append(f"Primary routing error: {str(e)}")
            
        return results
        
    async def route_read(self, nova_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """Route a read operation across appropriate layers"""
        # Determine which layers to query based on query parameters
        target_layers = query.get('layers', [])
        
        if not target_layers:
            # Auto-determine based on query
            if 'memory_type' in query:
                memory_type = MemoryType(query['memory_type'])
                if memory_type in self.TYPE_TO_LAYERS:
                    config = self.TYPE_TO_LAYERS[memory_type]
                    target_layers = [config['primary']] + config['secondary']
            else:
                # Default to working memory and recent layers
                target_layers = [3, 6, 7, 8, 9, 10]
                
        # Read from layers
        results = {
            'query': query,
            'results_by_layer': {},
            'merged_results': [],
            'total_count': 0
        }
        
        # Parallel reads
        tasks = []
        for layer_id in target_layers:
            if layer_id <= 10:
                layer = self.layer_managers['immediate'].layers[layer_id]
                tasks.append(layer.read(nova_id, query))
                
        if tasks:
            layer_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(layer_results):
                layer_id = target_layers[i]
                if isinstance(result, Exception):
                    results['results_by_layer'][layer_id] = {'error': str(result)}
                else:
                    results['results_by_layer'][layer_id] = {
                        'count': len(result),
                        'memories': [m.to_dict() for m in result]
                    }
                    results['merged_results'].extend(result)
                    results['total_count'] += len(result)
                    
        # Sort merged results by timestamp
        results['merged_results'].sort(
            key=lambda x: x.timestamp if hasattr(x, 'timestamp') else x.get('timestamp', ''),
            reverse=True
        )
        
        return results
        
    async def cross_layer_query(self, nova_id: str, query: str, 
                               layers: Optional[List[int]] = None) -> List[MemoryEntry]:
        """Execute a query across multiple layers"""
        # This would integrate with MeiliSearch for full-text search
        # For now, simple implementation
        
        if not layers:
            layers = list(range(1, 11))  # All immediate layers
            
        all_results = []
        
        for layer_id in layers:
            if layer_id <= 10:
                layer = self.layer_managers['immediate'].layers[layer_id]
                # Simple keyword search in data
                memories = await layer.read(nova_id)
                for memory in memories:
                    if query.lower() in json.dumps(memory.data).lower():
                        all_results.append(memory)
                        
        return all_results
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get router performance metrics"""
        return {
            **self.performance_metrics,
            'cache_size': len(self.routing_cache),
            'hit_rate': self.performance_metrics['cache_hits'] / max(self.performance_metrics['total_routes'], 1)
        }

# Example usage
async def test_memory_router():
    """Test memory router functionality"""
    
    # Initialize database pool
    db_pool = NovaDatabasePool()
    await db_pool.initialize_all_connections()
    
    # Create router
    router = MemoryRouter(db_pool)
    await router.initialize()
    
    # Test routing decisions
    test_memories = [
        {
            'content': 'User said hello',
            'importance': 0.7,
            'interaction': True
        },
        {
            'content': 'Need to complete task: respond to user',
            'task': 'respond',
            'importance': 0.8
        },
        {
            'content': 'Learned new concept: memory routing',
            'concept': 'memory routing',
            'knowledge': True
        }
    ]
    
    for memory in test_memories:
        # Get routing decision
        decision = router.get_routing_decision(memory)
        print(f"\nMemory: {memory['content']}")
        print(f"Primary Layer: {decision.primary_layer}")
        print(f"Secondary Layers: {decision.secondary_layers}")
        print(f"Databases: {decision.databases}")
        
        # Route write
        result = await router.route_write('bloom', memory)
        print(f"Write Result: {result['primary_result']}")
        
    # Test read
    read_result = await router.route_read('bloom', {'memory_type': 'task'})
    print(f"\nRead Results: {read_result['total_count']} memories found")
    
    # Performance metrics
    print(f"\nPerformance: {router.get_performance_metrics()}")
    
    # Cleanup
    await db_pool.close_all()

if __name__ == "__main__":
    asyncio.run(test_memory_router())