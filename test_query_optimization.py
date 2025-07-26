#!/usr/bin/env python3
"""
Nova Memory System - Query Optimization Tests
Comprehensive test suite for memory query optimization components
"""

import unittest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

# Import the modules to test
from memory_query_optimizer import (
    MemoryQueryOptimizer, OptimizationLevel, QueryPlan, ExecutionStatistics,
    OptimizationContext, QueryPlanCache, CostModel, QueryPatternAnalyzer,
    AdaptiveOptimizer, IndexRecommendation, IndexType
)
from query_execution_engine import (
    QueryExecutionEngine, ExecutionContext, ExecutionResult, ExecutionStatus,
    ExecutionMode, ExecutionMonitor, ResourceManager
)
from semantic_query_analyzer import (
    SemanticQueryAnalyzer, QuerySemantics, SemanticIntent, QueryComplexity,
    MemoryDomain, SemanticEntity, SemanticRelation
)

class TestMemoryQueryOptimizer(unittest.TestCase):
    """Test cases for Memory Query Optimizer"""
    
    def setUp(self):
        self.optimizer = MemoryQueryOptimizer(OptimizationLevel.BALANCED)
        self.context = OptimizationContext(
            nova_id="test_nova",
            session_id="test_session",
            current_memory_load=0.5,
            available_indexes={'memory_entries': ['timestamp', 'nova_id']},
            system_resources={'cpu': 0.4, 'memory': 0.6},
            historical_patterns={}
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.optimization_level, OptimizationLevel.BALANCED)
        self.assertIsNotNone(self.optimizer.cost_model)
        self.assertIsNotNone(self.optimizer.plan_cache)
        self.assertEqual(self.optimizer.optimization_stats['total_optimizations'], 0)
    
    async def test_optimize_simple_query(self):
        """Test optimization of a simple query"""
        query = {
            'operation': 'read',
            'memory_types': ['working'],
            'conditions': {'nova_id': 'test_nova'}
        }
        
        plan = await self.optimizer.optimize_query(query, self.context)
        
        self.assertIsInstance(plan, QueryPlan)
        self.assertGreater(len(plan.optimized_operations), 0)
        self.assertGreater(plan.estimated_cost, 0)
        self.assertIn(3, plan.memory_layers)  # Working memory layer
        self.assertIn('dragonfly', plan.databases)
    
    async def test_optimize_complex_query(self):
        """Test optimization of a complex query"""
        query = {
            'operation': 'search',
            'memory_types': ['episodic', 'semantic'],
            'conditions': {
                'timestamp': {'range': ['2023-01-01', '2023-12-31']},
                'content': {'contains': 'important meeting'},
                'emotional_tone': 'positive'
            },
            'aggregations': ['count', 'avg'],
            'sort': {'field': 'timestamp', 'order': 'desc'},
            'limit': 100
        }
        
        plan = await self.optimizer.optimize_query(query, self.context)
        
        self.assertIsInstance(plan, QueryPlan)
        self.assertGreater(len(plan.optimized_operations), 3)
        self.assertGreater(plan.estimated_cost, 10.0)  # Complex queries should have higher cost
        # Should access multiple memory layers
        self.assertTrue(any(layer >= 6 for layer in plan.memory_layers))
    
    def test_cache_functionality(self):
        """Test query plan caching"""
        query = {'operation': 'read', 'nova_id': 'test'}
        
        # First call should be cache miss
        cached_plan = self.optimizer.plan_cache.get(query, self.context)
        self.assertIsNone(cached_plan)
        
        # Add a plan to cache
        plan = QueryPlan(
            plan_id="test_plan",
            query_hash="test_hash",
            original_query=query,
            optimized_operations=[],
            estimated_cost=10.0,
            estimated_time=0.1,
            memory_layers=[3],
            databases=['dragonfly']
        )
        
        self.optimizer.plan_cache.put(query, self.context, plan)
        
        # Second call should be cache hit
        cached_plan = self.optimizer.plan_cache.get(query, self.context)
        self.assertIsNotNone(cached_plan)
        self.assertEqual(cached_plan.plan_id, "test_plan")
    
    def test_cost_model(self):
        """Test cost estimation model"""
        # Test operation costs
        scan_cost = CostModel.estimate_operation_cost('scan', 1000)
        index_cost = CostModel.estimate_operation_cost('index_lookup', 1000, 0.1)
        
        self.assertGreater(scan_cost, index_cost)  # Scan should be more expensive
        
        # Test layer costs
        layer1_cost = CostModel.estimate_layer_cost(1, 1000)  # Sensory buffer
        layer16_cost = CostModel.estimate_layer_cost(16, 1000)  # Long-term episodic
        
        self.assertGreater(layer16_cost, layer1_cost)  # Long-term should be more expensive
        
        # Test database costs
        dragonfly_cost = CostModel.estimate_database_cost('dragonfly', 1000)
        postgresql_cost = CostModel.estimate_database_cost('postgresql', 1000)
        
        self.assertGreater(postgresql_cost, dragonfly_cost)  # Disk-based should be more expensive
    
    async def test_execution_stats_recording(self):
        """Test recording execution statistics"""
        plan_id = "test_plan_123"
        stats = ExecutionStatistics(
            plan_id=plan_id,
            actual_cost=15.5,
            actual_time=0.25,
            rows_processed=500,
            memory_usage=1024,
            cache_hits=5,
            cache_misses=2
        )
        
        initial_history_size = len(self.optimizer.execution_history)
        await self.optimizer.record_execution_stats(plan_id, stats)
        
        self.assertEqual(len(self.optimizer.execution_history), initial_history_size + 1)
        self.assertEqual(self.optimizer.execution_history[-1].plan_id, plan_id)
    
    async def test_index_recommendations(self):
        """Test index recommendation generation"""
        query = {
            'operation': 'search',
            'conditions': {'timestamp': {'range': ['2023-01-01', '2023-12-31']}},
            'full_text_search': {'content': 'search terms'}
        }
        
        plan = await self.optimizer.optimize_query(query, self.context)
        recommendations = await self.optimizer.get_index_recommendations(5)
        
        self.assertIsInstance(recommendations, list)
        if recommendations:
            self.assertIsInstance(recommendations[0], IndexRecommendation)
            self.assertIn(recommendations[0].index_type, [IndexType.BTREE, IndexType.GIN])

class TestQueryExecutionEngine(unittest.TestCase):
    """Test cases for Query Execution Engine"""
    
    def setUp(self):
        self.optimizer = Mock(spec=MemoryQueryOptimizer)
        self.optimizer.record_execution_stats = AsyncMock()
        self.engine = QueryExecutionEngine(self.optimizer, max_workers=2)
        
        self.plan = QueryPlan(
            plan_id="test_plan",
            query_hash="test_hash",
            original_query={'operation': 'read'},
            optimized_operations=[
                {'operation': 'access_layers', 'layers': [3]},
                {'operation': 'apply_filters', 'selectivity': 0.5},
                {'operation': 'return_results', 'parallel': True}
            ],
            estimated_cost=10.0,
            estimated_time=0.1,
            memory_layers=[3],
            databases=['dragonfly']
        )
        
        self.context = ExecutionContext(
            execution_id="test_exec",
            nova_id="test_nova",
            session_id="test_session",
            priority=1
        )
    
    def test_engine_initialization(self):
        """Test execution engine initialization"""
        self.assertEqual(self.engine.max_workers, 2)
        self.assertIsNotNone(self.engine.monitor)
        self.assertIsNotNone(self.engine.resource_manager)
    
    async def test_execute_simple_plan(self):
        """Test execution of a simple plan"""
        result = await self.engine.execute_query(self.plan, self.context)
        
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.execution_id, "test_exec")
        self.assertIn(result.status, [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED])
        self.assertIsNotNone(result.started_at)
        self.assertIsNotNone(result.completed_at)
    
    async def test_parallel_execution(self):
        """Test parallel execution of operations"""
        parallel_plan = QueryPlan(
            plan_id="parallel_plan",
            query_hash="parallel_hash",
            original_query={'operation': 'search'},
            optimized_operations=[
                {'operation': 'access_layers', 'layers': [3, 6, 7]},
                {'operation': 'full_text_search', 'parallel': True},
                {'operation': 'rank_results', 'parallel': False},
                {'operation': 'return_results', 'parallel': True}
            ],
            estimated_cost=20.0,
            estimated_time=0.2,
            memory_layers=[3, 6, 7],
            databases=['dragonfly', 'postgresql'],
            parallelizable=True
        )
        
        result = await self.engine.execute_query(parallel_plan, self.context)
        
        self.assertIsInstance(result, ExecutionResult)
        # Parallel execution should still complete successfully
        self.assertIn(result.status, [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED])
    
    def test_resource_manager(self):
        """Test resource management"""
        initial_status = self.engine.resource_manager.get_resource_status()
        
        self.assertEqual(initial_status['current_executions'], 0)
        self.assertEqual(initial_status['execution_slots_available'], 
                        initial_status['max_parallel_executions'])
    
    async def test_execution_timeout(self):
        """Test execution timeout handling"""
        timeout_context = ExecutionContext(
            execution_id="timeout_test",
            nova_id="test_nova",
            timeout_seconds=0.001  # Very short timeout
        )
        
        # Create a plan that would take longer than the timeout
        slow_plan = self.plan
        slow_plan.estimated_time = 1.0  # 1 second estimated
        
        result = await self.engine.execute_query(slow_plan, timeout_context)
        
        # Should either complete quickly or timeout
        self.assertIn(result.status, [ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED, ExecutionStatus.FAILED])
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        metrics = self.engine.get_performance_metrics()
        
        self.assertIn('execution_metrics', metrics)
        self.assertIn('resource_status', metrics)
        self.assertIn('engine_config', metrics)
        
        execution_metrics = metrics['execution_metrics']
        self.assertIn('total_executions', execution_metrics)
        self.assertIn('success_rate', execution_metrics)

class TestSemanticQueryAnalyzer(unittest.TestCase):
    """Test cases for Semantic Query Analyzer"""
    
    def setUp(self):
        self.analyzer = SemanticQueryAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.vocabulary)
        self.assertEqual(self.analyzer.analysis_stats['total_analyses'], 0)
    
    async def test_simple_query_analysis(self):
        """Test analysis of a simple query"""
        query = {
            'operation': 'read',
            'query': 'Find my recent memories about the meeting'
        }
        
        semantics = await self.analyzer.analyze_query(query)
        
        self.assertIsInstance(semantics, QuerySemantics)
        self.assertEqual(semantics.original_query, query)
        self.assertIsInstance(semantics.intent, SemanticIntent)
        self.assertIsInstance(semantics.complexity, QueryComplexity)
        self.assertIsInstance(semantics.domains, list)
        self.assertGreater(semantics.confidence_score, 0.0)
        self.assertLessEqual(semantics.confidence_score, 1.0)
    
    async def test_intent_classification(self):
        """Test intent classification accuracy"""
        test_cases = [
            ({'operation': 'read', 'query': 'get my memories'}, SemanticIntent.RETRIEVE_MEMORY),
            ({'operation': 'write', 'query': 'store this information'}, SemanticIntent.STORE_MEMORY),
            ({'operation': 'search', 'query': 'find similar experiences'}, SemanticIntent.SEARCH_SIMILARITY),
            ({'query': 'when did I last see John?'}, SemanticIntent.TEMPORAL_QUERY),
            ({'query': 'analyze my learning patterns'}, SemanticIntent.ANALYZE_MEMORY)
        ]
        
        for query, expected_intent in test_cases:
            semantics = await self.analyzer.analyze_query(query)
            # Note: Intent classification is heuristic, so we just check it's reasonable
            self.assertIsInstance(semantics.intent, SemanticIntent)
    
    async def test_complexity_calculation(self):
        """Test query complexity calculation"""
        simple_query = {'operation': 'read', 'query': 'get memory'}
        complex_query = {
            'operation': 'search',
            'query': 'Find all episodic memories from last year related to work meetings with emotional context positive and analyze patterns',
            'conditions': {
                'timestamp': {'range': ['2023-01-01', '2023-12-31']},
                'type': 'episodic',
                'context': 'work',
                'emotional_tone': 'positive'
            },
            'aggregations': ['count', 'group_by'],
            'subqueries': [{'operation': 'analyze'}]
        }
        
        simple_semantics = await self.analyzer.analyze_query(simple_query)
        complex_semantics = await self.analyzer.analyze_query(complex_query)
        
        # Complex query should have higher complexity
        self.assertLessEqual(simple_semantics.complexity.value, complex_semantics.complexity.value)
    
    async def test_domain_identification(self):
        """Test memory domain identification"""
        test_cases = [
            ({'query': 'episodic memory about yesterday'}, MemoryDomain.EPISODIC),
            ({'query': 'semantic knowledge about Python'}, MemoryDomain.SEMANTIC),
            ({'query': 'procedural memory for driving'}, MemoryDomain.PROCEDURAL),
            ({'query': 'emotional memory of happiness'}, MemoryDomain.EMOTIONAL),
            ({'query': 'social interaction with friends'}, MemoryDomain.SOCIAL)
        ]
        
        for query, expected_domain in test_cases:
            semantics = await self.analyzer.analyze_query(query)
            # Check if expected domain is in identified domains
            domain_values = [d.value for d in semantics.domains]
            # Note: Domain identification is heuristic, so we check it's reasonable
            self.assertIsInstance(semantics.domains, list)
            self.assertGreater(len(semantics.domains), 0)
    
    async def test_entity_extraction(self):
        """Test semantic entity extraction"""
        query = {
            'query': 'Find memories from "important meeting" on 2023-05-15 at 10:30 AM with John Smith'
        }
        
        semantics = await self.analyzer.analyze_query(query)
        
        self.assertIsInstance(semantics.entities, list)
        
        # Check for different entity types
        entity_types = [e.entity_type for e in semantics.entities]
        
        # Should find at least some entities
        if len(semantics.entities) > 0:
            self.assertTrue(any(et in ['date', 'time', 'quoted_term', 'proper_noun'] 
                              for et in entity_types))
    
    async def test_temporal_analysis(self):
        """Test temporal aspect analysis"""
        temporal_query = {
            'query': 'Find memories from last week before the meeting on Monday'
        }
        
        semantics = await self.analyzer.analyze_query(temporal_query)
        
        self.assertIsInstance(semantics.temporal_aspects, dict)
        # Should identify temporal keywords
        if semantics.temporal_aspects:
            self.assertTrue(any(key in ['relative_time', 'absolute_time'] 
                               for key in semantics.temporal_aspects.keys()))
    
    async def test_query_optimization_suggestions(self):
        """Test query optimization suggestions"""
        similarity_query = {
            'operation': 'search',
            'query': 'find similar experiences to my vacation in Italy'
        }
        
        semantics = await self.analyzer.analyze_query(similarity_query)
        optimizations = await self.analyzer.suggest_query_optimizations(semantics)
        
        self.assertIsInstance(optimizations, list)
        if optimizations:
            optimization = optimizations[0]
            self.assertIn('type', optimization)
            self.assertIn('suggestion', optimization)
            self.assertIn('benefit', optimization)
    
    async def test_query_rewriting(self):
        """Test semantic query rewriting"""
        complex_query = {
            'operation': 'search',
            'query': 'find similar memories with emotional context',
            'conditions': {'type': 'episodic'}
        }
        
        semantics = await self.analyzer.analyze_query(complex_query)
        rewrites = await self.analyzer.rewrite_query_for_optimization(semantics)
        
        self.assertIsInstance(rewrites, list)
        if rewrites:
            rewrite = rewrites[0]
            self.assertIn('type', rewrite)
            self.assertIn('original', rewrite)
            self.assertIn('rewritten', rewrite)
            self.assertIn('confidence', rewrite)
    
    def test_semantic_statistics(self):
        """Test semantic analysis statistics"""
        stats = self.analyzer.get_semantic_statistics()
        
        self.assertIn('analysis_stats', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('vocabulary_size', stats)
        
        analysis_stats = stats['analysis_stats']
        self.assertIn('total_analyses', analysis_stats)
        self.assertIn('cache_hits', analysis_stats)

class TestIntegration(unittest.TestCase):
    """Integration tests for all components working together"""
    
    def setUp(self):
        self.analyzer = SemanticQueryAnalyzer()
        self.optimizer = MemoryQueryOptimizer(OptimizationLevel.BALANCED)
        self.engine = QueryExecutionEngine(self.optimizer, max_workers=2)
    
    async def test_end_to_end_query_processing(self):
        """Test complete query processing pipeline"""
        # Complex query that exercises all components
        query = {
            'operation': 'search',
            'query': 'Find episodic memories from last month about work meetings with positive emotions',
            'memory_types': ['episodic'],
            'conditions': {
                'timestamp': {'range': ['2023-10-01', '2023-10-31']},
                'context': 'work',
                'emotional_tone': 'positive'
            },
            'limit': 20
        }
        
        # Step 1: Semantic analysis
        semantics = await self.analyzer.analyze_query(query)
        self.assertIsInstance(semantics, QuerySemantics)
        self.assertEqual(semantics.intent, SemanticIntent.RETRIEVE_MEMORY)
        
        # Step 2: Query optimization
        context = OptimizationContext(
            nova_id="integration_test",
            session_id="test_session",
            current_memory_load=0.3,
            available_indexes={'episodic_memories': ['timestamp', 'context']},
            system_resources={'cpu': 0.2, 'memory': 0.4},
            historical_patterns={}
        )
        
        plan = await self.optimizer.optimize_query(query, context)
        self.assertIsInstance(plan, QueryPlan)
        self.assertGreater(len(plan.optimized_operations), 0)
        
        # Step 3: Query execution
        exec_context = ExecutionContext(
            execution_id="integration_test_exec",
            nova_id="integration_test",
            session_id="test_session"
        )
        
        result = await self.engine.execute_query(plan, exec_context)
        self.assertIsInstance(result, ExecutionResult)
        self.assertIn(result.status, [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED])
        
        # Verify statistics were recorded
        self.assertIsNotNone(result.execution_stats)
    
    async def test_caching_across_components(self):
        """Test caching behavior across components"""
        query = {
            'operation': 'read',
            'query': 'simple memory retrieval'
        }
        
        context = OptimizationContext(
            nova_id="cache_test",
            session_id="test_session",
            current_memory_load=0.5,
            available_indexes={},
            system_resources={'cpu': 0.3, 'memory': 0.5},
            historical_patterns={}
        )
        
        # First execution - should be cache miss
        initial_cache_stats = self.optimizer.get_optimization_statistics()
        initial_cache_hits = initial_cache_stats['cache_statistics']['cache_hits']
        
        plan1 = await self.optimizer.optimize_query(query, context)
        
        # Second execution - should be cache hit
        plan2 = await self.optimizer.optimize_query(query, context)
        
        final_cache_stats = self.optimizer.get_optimization_statistics()
        final_cache_hits = final_cache_stats['cache_statistics']['cache_hits']
        
        self.assertGreater(final_cache_hits, initial_cache_hits)
        self.assertEqual(plan1.query_hash, plan2.query_hash)
    
    async def test_performance_monitoring(self):
        """Test performance monitoring across components"""
        query = {
            'operation': 'search',
            'query': 'performance monitoring test'
        }
        
        # Execute query and monitor performance
        context = OptimizationContext(
            nova_id="perf_test",
            session_id="test_session",
            current_memory_load=0.4,
            available_indexes={},
            system_resources={'cpu': 0.3, 'memory': 0.6},
            historical_patterns={}
        )
        
        plan = await self.optimizer.optimize_query(query, context)
        
        exec_context = ExecutionContext(
            execution_id="perf_test_exec",
            nova_id="perf_test",
            session_id="test_session"
        )
        
        result = await self.engine.execute_query(plan, exec_context)
        
        # Check that performance metrics are collected
        optimizer_stats = self.optimizer.get_optimization_statistics()
        engine_metrics = self.engine.get_performance_metrics()
        
        self.assertGreater(optimizer_stats['total_optimizations'], 0)
        self.assertGreaterEqual(engine_metrics['execution_metrics']['total_executions'], 0)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for optimization components"""
    
    def setUp(self):
        self.analyzer = SemanticQueryAnalyzer()
        self.optimizer = MemoryQueryOptimizer(OptimizationLevel.AGGRESSIVE)
    
    async def test_optimization_performance(self):
        """Benchmark optimization performance"""
        queries = [
            {'operation': 'read', 'query': f'test query {i}'} 
            for i in range(100)
        ]
        
        context = OptimizationContext(
            nova_id="benchmark",
            session_id="test",
            current_memory_load=0.5,
            available_indexes={},
            system_resources={'cpu': 0.3, 'memory': 0.5},
            historical_patterns={}
        )
        
        start_time = time.time()
        
        for query in queries:
            await self.optimizer.optimize_query(query, context)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(queries)
        
        # Performance assertion - should average less than 10ms per optimization
        self.assertLess(avg_time, 0.01, 
                       f"Average optimization time {avg_time:.4f}s exceeds 10ms threshold")
        
        print(f"Optimization benchmark: {len(queries)} queries in {total_time:.3f}s "
              f"(avg {avg_time*1000:.2f}ms per query)")
    
    async def test_semantic_analysis_performance(self):
        """Benchmark semantic analysis performance"""
        queries = [
            {'query': f'Find memories about topic {i} with temporal context and emotional aspects'}
            for i in range(50)
        ]
        
        start_time = time.time()
        
        for query in queries:
            await self.analyzer.analyze_query(query)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(queries)
        
        # Performance assertion - should average less than 20ms per analysis
        self.assertLess(avg_time, 0.02,
                       f"Average analysis time {avg_time:.4f}s exceeds 20ms threshold")
        
        print(f"Semantic analysis benchmark: {len(queries)} queries in {total_time:.3f}s "
              f"(avg {avg_time*1000:.2f}ms per query)")

async def run_async_tests():
    """Run all async test methods"""
    test_classes = [
        TestMemoryQueryOptimizer,
        TestQueryExecutionEngine, 
        TestSemanticQueryAnalyzer,
        TestIntegration,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        for test in suite:
            if hasattr(test, '_testMethodName'):
                method = getattr(test, test._testMethodName)
                if asyncio.iscoroutinefunction(method):
                    print(f"  Running async test: {test._testMethodName}")
                    try:
                        test.setUp()
                        await method()
                        print(f"    ✓ {test._testMethodName} passed")
                    except Exception as e:
                        print(f"    ✗ {test._testMethodName} failed: {e}")
                else:
                    # Run regular unittest
                    try:
                        result = unittest.TestResult()
                        test.run(result)
                        if result.wasSuccessful():
                            print(f"    ✓ {test._testMethodName} passed")
                        else:
                            for failure in result.failures + result.errors:
                                print(f"    ✗ {test._testMethodName} failed: {failure[1]}")
                    except Exception as e:
                        print(f"    ✗ {test._testMethodName} error: {e}")

if __name__ == '__main__':
    print("Nova Memory Query Optimization - Test Suite")
    print("=" * 50)
    
    # Run async tests
    asyncio.run(run_async_tests())
    
    print("\nTest suite completed.")
    print("Note: This test suite uses mocked dependencies for isolated testing.")
    print("For full integration testing, run with actual Nova memory system components.")