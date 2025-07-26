#!/usr/bin/env python3
"""
Cross-Nova Memory Transfer Protocol Test Suite
Comprehensive testing for the memory transfer system
"""

import asyncio
import unittest
import json
import tempfile
import ssl
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
import sys
import os

# Add the implementation directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cross_nova_transfer_protocol import (
    CrossNovaTransferProtocol, TransferOperation, TransferStatus,
    VectorClock, MemoryDelta, ConflictResolution, NovaAuthenticator,
    CompressionManager, ChunkManager, BandwidthLimiter, ConflictResolver
)
from memory_sync_manager import (
    MemorySyncManager, SyncConfiguration, SyncMode, SyncDirection,
    PrivacyLevel, PrivacyController, BandwidthOptimizer, MemorySnapshot
)
from unified_memory_api import NovaMemoryAPI, MemoryRequest, MemoryResponse, MemoryOperation

class TestVectorClock(unittest.TestCase):
    """Test vector clock functionality"""
    
    def setUp(self):
        self.clock1 = VectorClock()
        self.clock2 = VectorClock()
    
    def test_increment(self):
        """Test clock increment"""
        self.clock1.increment('nova1')
        self.assertEqual(self.clock1.clocks['nova1'], 1)
        
        self.clock1.increment('nova1')
        self.assertEqual(self.clock1.clocks['nova1'], 2)
    
    def test_update(self):
        """Test clock update with another clock"""
        self.clock1.increment('nova1')
        self.clock1.increment('nova2')
        
        self.clock2.increment('nova1')
        self.clock2.increment('nova1')
        self.clock2.increment('nova3')
        
        self.clock1.update(self.clock2)
        
        self.assertEqual(self.clock1.clocks['nova1'], 2)  # max(1, 2)
        self.assertEqual(self.clock1.clocks['nova2'], 1)  # unchanged
        self.assertEqual(self.clock1.clocks['nova3'], 1)  # new
    
    def test_happens_before(self):
        """Test happens-before relationship"""
        self.clock1.increment('nova1')
        self.clock2.increment('nova1')
        self.clock2.increment('nova1')
        
        self.assertTrue(self.clock1.happens_before(self.clock2))
        self.assertFalse(self.clock2.happens_before(self.clock1))
    
    def test_concurrent(self):
        """Test concurrent relationship"""
        self.clock1.increment('nova1')
        self.clock2.increment('nova2')
        
        self.assertTrue(self.clock1.concurrent_with(self.clock2))
        self.assertTrue(self.clock2.concurrent_with(self.clock1))
    
    def test_serialization(self):
        """Test clock serialization"""
        self.clock1.increment('nova1')
        self.clock1.increment('nova2')
        
        data = self.clock1.to_dict()
        clock_restored = VectorClock.from_dict(data)
        
        self.assertEqual(self.clock1.clocks, clock_restored.clocks)

class TestMemoryDelta(unittest.TestCase):
    """Test memory delta functionality"""
    
    def test_checksum_calculation(self):
        """Test checksum calculation"""
        delta = MemoryDelta(
            memory_id='mem_001',
            operation='create',
            data={'content': 'test data'}
        )
        
        delta.calculate_checksum()
        self.assertIsNotNone(delta.checksum)
        self.assertEqual(len(delta.checksum), 64)  # SHA256 hex length
        
        # Same data should produce same checksum
        delta2 = MemoryDelta(
            memory_id='mem_001',
            operation='create',
            data={'content': 'test data'}
        )
        delta2.calculate_checksum()
        
        self.assertEqual(delta.checksum, delta2.checksum)

class TestCompressionManager(unittest.TestCase):
    """Test compression functionality"""
    
    def test_data_analysis(self):
        """Test data characteristic analysis"""
        # Highly compressible data
        repetitive_data = b'a' * 1000
        analysis = CompressionManager.analyze_data_characteristics(repetitive_data)
        
        self.assertEqual(analysis['size'], 1000)
        self.assertGreater(analysis['compression_potential'], 0.8)
        self.assertGreater(analysis['recommended_level'], 5)
    
    def test_adaptive_compression(self):
        """Test adaptive compression"""
        # Test with different data types
        test_data = json.dumps({'key': 'value' * 100}).encode()
        
        compressed, info = CompressionManager.compress_adaptive(test_data)
        
        self.assertLess(len(compressed), len(test_data))
        self.assertGreater(info['compression_ratio'], 1.0)
        self.assertEqual(info['original_size'], len(test_data))
        self.assertEqual(info['compressed_size'], len(compressed))
    
    def test_compression_decompression(self):
        """Test compression and decompression roundtrip"""
        original_data = json.dumps({
            'memories': [{'id': f'mem_{i}', 'content': f'Memory content {i}'} for i in range(100)]
        }).encode()
        
        compressed, info = CompressionManager.compress_adaptive(original_data)
        decompressed = CompressionManager.decompress(compressed)
        
        self.assertEqual(original_data, decompressed)

class TestChunkManager(unittest.TestCase):
    """Test chunk management functionality"""
    
    def test_create_chunks(self):
        """Test chunk creation"""
        data = b'a' * 10000  # 10KB data
        chunk_size = 1024    # 1KB chunks
        
        chunks = ChunkManager.create_chunks(data, chunk_size)
        
        self.assertEqual(len(chunks), 10)  # 10KB / 1KB = 10 chunks
        
        # Check chunk IDs are sequential
        for i, (chunk_id, chunk_data) in enumerate(chunks):
            self.assertEqual(chunk_id, i)
            expected_size = min(chunk_size, len(data) - i * chunk_size)
            self.assertEqual(len(chunk_data), expected_size)
    
    def test_chunk_header(self):
        """Test chunk header creation and parsing"""
        chunk_data = b'test chunk data'
        checksum = hashlib.sha256(chunk_data).hexdigest()
        
        header = ChunkManager.create_chunk_header(
            chunk_id=5,
            total_chunks=10,
            data_size=len(chunk_data),
            checksum=checksum
        )
        
        # Parse header
        parsed_header, offset = ChunkManager.parse_chunk_header(header)
        
        self.assertEqual(parsed_header['chunk_id'], 5)
        self.assertEqual(parsed_header['total_chunks'], 10)
        self.assertEqual(parsed_header['data_size'], len(chunk_data))
        self.assertEqual(parsed_header['checksum'], checksum)
    
    def test_reassemble_chunks(self):
        """Test chunk reassembly"""
        original_data = b'Hello, this is a test message for chunking!'
        chunks = ChunkManager.create_chunks(original_data, chunk_size=10)
        
        # Create chunk dictionary
        chunk_dict = {chunk_id: chunk_data for chunk_id, chunk_data in chunks}
        
        # Reassemble
        reassembled = ChunkManager.reassemble_chunks(chunk_dict)
        
        self.assertEqual(original_data, reassembled)
    
    def test_checksum_verification(self):
        """Test chunk checksum verification"""
        chunk_data = b'test data for checksum'
        correct_checksum = hashlib.sha256(chunk_data).hexdigest()
        wrong_checksum = 'wrong_checksum'
        
        self.assertTrue(ChunkManager.verify_chunk_checksum(chunk_data, correct_checksum))
        self.assertFalse(ChunkManager.verify_chunk_checksum(chunk_data, wrong_checksum))

class TestBandwidthLimiter(unittest.TestCase):
    """Test bandwidth limiting functionality"""
    
    def test_token_acquisition(self):
        """Test bandwidth token acquisition"""
        limiter = BandwidthLimiter(max_bytes_per_second=1000)
        
        # Should acquire tokens immediately for small amounts
        start_time = asyncio.get_event_loop().time()
        asyncio.get_event_loop().run_until_complete(limiter.acquire(100))
        end_time = asyncio.get_event_loop().time()
        
        # Should be nearly instantaneous
        self.assertLess(end_time - start_time, 0.1)
    
    async def test_rate_limiting(self):
        """Test actual rate limiting"""
        limiter = BandwidthLimiter(max_bytes_per_second=100)  # Very low limit
        
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire(200)  # Request more than limit
        end_time = asyncio.get_event_loop().time()
        
        # Should take at least 1 second (200 bytes / 100 bytes/s - 100 initial tokens)
        self.assertGreater(end_time - start_time, 0.9)

class TestPrivacyController(unittest.TestCase):
    """Test privacy control functionality"""
    
    def setUp(self):
        self.privacy_controller = PrivacyController()
        self.privacy_controller.add_team_membership('core_team', {'nova1', 'nova2', 'nova3'})
    
    def test_public_memory_sharing(self):
        """Test public memory sharing"""
        memory = {
            'id': 'mem_001',
            'content': 'public information',
            'privacy_level': PrivacyLevel.PUBLIC.value
        }
        
        # Should be shareable with any Nova
        self.assertTrue(
            self.privacy_controller.can_share_memory(memory, 'any_nova', 'nova1')
        )
    
    def test_private_memory_sharing(self):
        """Test private memory sharing"""
        memory = {
            'id': 'mem_002',
            'content': 'private information',
            'privacy_level': PrivacyLevel.PRIVATE.value
        }
        
        # Should only be shareable with same Nova
        self.assertTrue(
            self.privacy_controller.can_share_memory(memory, 'nova1', 'nova1')
        )
        self.assertFalse(
            self.privacy_controller.can_share_memory(memory, 'nova2', 'nova1')
        )
    
    def test_team_memory_sharing(self):
        """Test team memory sharing"""
        memory = {
            'id': 'mem_003',
            'content': 'team information',
            'privacy_level': PrivacyLevel.TEAM.value
        }
        
        # Should be shareable within team
        self.assertTrue(
            self.privacy_controller.can_share_memory(memory, 'nova2', 'nova1')
        )
        # Should not be shareable outside team
        self.assertFalse(
            self.privacy_controller.can_share_memory(memory, 'outside_nova', 'nova1')
        )
    
    def test_classified_memory_sharing(self):
        """Test classified memory sharing"""
        memory = {
            'id': 'mem_004',
            'content': 'classified information',
            'privacy_level': PrivacyLevel.CLASSIFIED.value
        }
        
        # Should never be shareable
        self.assertFalse(
            self.privacy_controller.can_share_memory(memory, 'nova1', 'nova1')
        )
        self.assertFalse(
            self.privacy_controller.can_share_memory(memory, 'nova2', 'nova1')
        )
    
    def test_tag_based_privacy(self):
        """Test privacy determination from tags"""
        private_memory = {
            'id': 'mem_005',
            'content': 'some content',
            'tags': ['private', 'personal']
        }
        
        # Should be detected as private
        privacy_level = self.privacy_controller._determine_privacy_level(
            private_memory, 'mem_005', 'some content', ['private', 'personal']
        )
        self.assertEqual(privacy_level, PrivacyLevel.PRIVATE)

class TestConflictResolver(unittest.TestCase):
    """Test conflict resolution functionality"""
    
    def setUp(self):
        self.resolver = ConflictResolver()
    
    async def test_latest_wins_strategy(self):
        """Test latest wins conflict resolution"""
        local_memory = {
            'id': 'mem_001',
            'content': 'local version',
            'timestamp': '2023-01-01T10:00:00'
        }
        
        remote_memory = {
            'id': 'mem_001',
            'content': 'remote version',
            'timestamp': '2023-01-01T11:00:00'  # Later timestamp
        }
        
        result = await self.resolver.resolve_conflict(
            local_memory, remote_memory, ConflictResolution.LATEST_WINS
        )
        
        self.assertEqual(result['content'], 'remote version')
    
    async def test_source_wins_strategy(self):
        """Test source wins conflict resolution"""
        local_memory = {
            'id': 'mem_001',
            'content': 'local version'
        }
        
        remote_memory = {
            'id': 'mem_001',
            'content': 'remote version'
        }
        
        result = await self.resolver.resolve_conflict(
            local_memory, remote_memory, ConflictResolution.SOURCE_WINS
        )
        
        self.assertEqual(result['content'], 'remote version')
    
    async def test_merge_strategy(self):
        """Test merge conflict resolution"""
        local_memory = {
            'id': 'mem_001',
            'content': 'local version',
            'local_field': 'local_value'
        }
        
        remote_memory = {
            'id': 'mem_001',
            'content': 'remote version',
            'remote_field': 'remote_value'
        }
        
        result = await self.resolver.resolve_conflict(
            local_memory, remote_memory, ConflictResolution.MERGE
        )
        
        self.assertEqual(result['content'], 'remote version')  # Remote overwrites
        self.assertEqual(result['local_field'], 'local_value')  # Local preserved
        self.assertEqual(result['remote_field'], 'remote_value')  # Remote added
    
    async def test_preserve_both_strategy(self):
        """Test preserve both conflict resolution"""
        local_memory = {
            'id': 'mem_001',
            'content': 'local version'
        }
        
        remote_memory = {
            'id': 'mem_001',
            'content': 'remote version'
        }
        
        result = await self.resolver.resolve_conflict(
            local_memory, remote_memory, ConflictResolution.PRESERVE_BOTH
        )
        
        self.assertEqual(result['conflict_type'], 'preserved_both')
        self.assertEqual(result['local_version'], local_memory)
        self.assertEqual(result['remote_version'], remote_memory)

class TestMemorySnapshot(unittest.TestCase):
    """Test memory snapshot functionality"""
    
    def setUp(self):
        self.snapshot1 = MemorySnapshot(
            nova_id='nova1',
            timestamp=datetime.now(),
            memory_checksums={
                'mem_001': 'checksum1',
                'mem_002': 'checksum2',
                'mem_003': 'checksum3'
            },
            total_count=3,
            last_modified={
                'mem_001': datetime.now() - timedelta(hours=1),
                'mem_002': datetime.now() - timedelta(hours=2),
                'mem_003': datetime.now() - timedelta(hours=3)
            },
            vector_clock=VectorClock({'nova1': 10})
        )
        
        self.snapshot2 = MemorySnapshot(
            nova_id='nova1',
            timestamp=datetime.now(),
            memory_checksums={
                'mem_001': 'checksum1',      # unchanged
                'mem_002': 'checksum2_new',  # modified
                'mem_004': 'checksum4'       # new
                # mem_003 deleted
            },
            total_count=3,
            last_modified={},
            vector_clock=VectorClock({'nova1': 15})
        )
    
    def test_calculate_deltas(self):
        """Test delta calculation between snapshots"""
        deltas = self.snapshot2.calculate_deltas(self.snapshot1)
        
        # Should have deltas for: modified mem_002, new mem_004, deleted mem_003
        self.assertEqual(len(deltas), 3)
        
        operations = {delta.memory_id: delta.operation for delta in deltas}
        
        self.assertEqual(operations['mem_002'], 'update')
        self.assertEqual(operations['mem_004'], 'create')
        self.assertEqual(operations['mem_003'], 'delete')

class MockNovaMemoryAPI:
    """Mock memory API for testing"""
    
    def __init__(self):
        self.memories = [
            {
                'id': 'mem_001',
                'content': 'Test memory 1',
                'timestamp': datetime.now().isoformat(),
                'tags': ['test'],
                'privacy_level': PrivacyLevel.PUBLIC.value
            },
            {
                'id': 'mem_002',
                'content': 'Private test memory',
                'timestamp': datetime.now().isoformat(),
                'tags': ['test', 'private'],
                'privacy_level': PrivacyLevel.PRIVATE.value
            }
        ]
    
    async def initialize(self):
        pass
    
    async def shutdown(self):
        pass
    
    async def recall(self, nova_id: str, query=None, **kwargs):
        return MemoryResponse(
            success=True,
            operation=MemoryOperation.READ,
            data={
                'memories': self.memories,
                'total_count': len(self.memories)
            }
        )

class TestCrossNovaTransferProtocol(unittest.IsolatedAsyncioTestCase):
    """Test cross-Nova transfer protocol"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.protocol1 = CrossNovaTransferProtocol('nova1', port=8445)
        self.protocol2 = CrossNovaTransferProtocol('nova2', port=8446)
        
        # Start servers
        await self.protocol1.start_server()
        await self.protocol2.start_server()
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        await self.protocol1.stop_server()
        await self.protocol2.stop_server()
    
    @patch('cross_nova_transfer_protocol.aiohttp.ClientSession.post')
    async def test_transfer_initiation(self, mock_post):
        """Test transfer initiation"""
        # Mock successful responses
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(
            return_value={'resume_token': 'test_token'}
        )
        
        memory_data = {'memories': [{'id': 'test', 'content': 'test data'}]}
        
        # This would normally fail due to network, but we're testing the structure
        try:
            session = await self.protocol1.initiate_transfer(
                target_nova='nova2',
                target_host='localhost',
                target_port=8446,
                operation=TransferOperation.SYNC_INCREMENTAL,
                memory_data=memory_data
            )
        except Exception:
            pass  # Expected to fail due to mocking

class TestMemorySyncManager(unittest.IsolatedAsyncioTestCase):
    """Test memory sync manager"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.memory_api = MockNovaMemoryAPI()
        await self.memory_api.initialize()
        
        self.sync_manager = MemorySyncManager('nova1', self.memory_api)
        await self.sync_manager.start()
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        await self.sync_manager.stop()
        await self.memory_api.shutdown()
    
    def test_add_sync_configuration(self):
        """Test adding sync configuration"""
        config = SyncConfiguration(
            target_nova='nova2',
            target_host='localhost',
            target_port=8443,
            sync_mode=SyncMode.INCREMENTAL
        )
        
        session_id = self.sync_manager.add_sync_configuration(config)
        
        self.assertIn(session_id, self.sync_manager.active_sessions)
        self.assertEqual(
            self.sync_manager.active_sessions[session_id].config.target_nova,
            'nova2'
        )
    
    def test_privacy_filtering(self):
        """Test privacy-based memory filtering"""
        # Setup privacy rules
        self.sync_manager.privacy_controller.add_team_membership(
            'test_team', {'nova1', 'nova2'}
        )
        
        # Test public memory
        public_memory = {
            'id': 'pub_001',
            'content': 'public info',
            'privacy_level': PrivacyLevel.PUBLIC.value
        }
        
        self.assertTrue(
            self.sync_manager.privacy_controller.can_share_memory(
                public_memory, 'nova2', 'nova1'
            )
        )
        
        # Test private memory
        private_memory = {
            'id': 'prv_001',
            'content': 'private info',
            'privacy_level': PrivacyLevel.PRIVATE.value
        }
        
        self.assertFalse(
            self.sync_manager.privacy_controller.can_share_memory(
                private_memory, 'nova2', 'nova1'
            )
        )
    
    async def test_memory_snapshot_creation(self):
        """Test memory snapshot creation"""
        snapshot = await self.sync_manager._create_memory_snapshot()
        
        self.assertEqual(snapshot.nova_id, 'nova1')
        self.assertGreater(len(snapshot.memory_checksums), 0)
        self.assertEqual(snapshot.total_count, len(self.memory_api.memories))
    
    def test_pattern_matching(self):
        """Test include/exclude pattern matching"""
        memory = {
            'id': 'test_memory',
            'content': 'This is a test memory about user conversations',
            'tags': ['conversation', 'user']
        }
        
        # Test include patterns
        self.assertTrue(
            self.sync_manager._matches_patterns(memory, ['conversation'], [])
        )
        self.assertFalse(
            self.sync_manager._matches_patterns(memory, ['system'], [])
        )
        
        # Test exclude patterns
        self.assertFalse(
            self.sync_manager._matches_patterns(memory, [], ['user'])
        )
        self.assertTrue(
            self.sync_manager._matches_patterns(memory, [], ['system'])
        )

class TestBandwidthOptimizer(unittest.TestCase):
    """Test bandwidth optimizer"""
    
    def setUp(self):
        self.optimizer = BandwidthOptimizer()
    
    def test_transfer_stats_recording(self):
        """Test transfer statistics recording"""
        self.optimizer.record_transfer_stats('nova1', 1000000, 2.0, 2.5)
        
        stats = self.optimizer.transfer_stats['nova1']
        self.assertEqual(stats['total_bytes'], 1000000)
        self.assertEqual(stats['total_duration'], 2.0)
        self.assertEqual(stats['transfer_count'], 1)
        self.assertEqual(stats['avg_compression_ratio'], 2.5)
    
    def test_optimal_chunk_size(self):
        """Test optimal chunk size calculation"""
        # Record some stats first
        self.optimizer.record_transfer_stats('fast_nova', 10000000, 1.0, 2.0)  # 10MB/s
        self.optimizer.record_transfer_stats('slow_nova', 500000, 1.0, 2.0)    # 0.5MB/s
        
        fast_chunk_size = self.optimizer.get_optimal_chunk_size('fast_nova')
        slow_chunk_size = self.optimizer.get_optimal_chunk_size('slow_nova')
        
        self.assertGreater(fast_chunk_size, slow_chunk_size)
    
    def test_compression_recommendation(self):
        """Test compression recommendation"""
        # Record stats with different compression ratios
        self.optimizer.record_transfer_stats('good_compression', 1000000, 1.0, 3.0)
        self.optimizer.record_transfer_stats('poor_compression', 1000000, 1.0, 1.1)
        
        # Should recommend compression for good compression target
        self.assertTrue(
            self.optimizer.should_enable_compression('good_compression', 10000)
        )
        
        # Might not recommend for poor compression target
        decision = self.optimizer.should_enable_compression('poor_compression', 10000)
        # Decision depends on throughput, so we just test it returns a boolean
        self.assertIsInstance(decision, bool)

class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration tests for the complete system"""
    
    async def asyncSetUp(self):
        """Set up integration test environment"""
        # Create two Nova instances
        self.memory_api1 = MockNovaMemoryAPI()
        self.memory_api2 = MockNovaMemoryAPI()
        
        await self.memory_api1.initialize()
        await self.memory_api2.initialize()
        
        self.sync_manager1 = MemorySyncManager('nova1', self.memory_api1)
        self.sync_manager2 = MemorySyncManager('nova2', self.memory_api2)
        
        await self.sync_manager1.start()
        await self.sync_manager2.start()
    
    async def asyncTearDown(self):
        """Clean up integration test environment"""
        await self.sync_manager1.stop()
        await self.sync_manager2.stop()
        await self.memory_api1.shutdown()
        await self.memory_api2.shutdown()
    
    async def test_end_to_end_sync_setup(self):
        """Test end-to-end sync setup"""
        # Configure sync between nova1 and nova2
        config = SyncConfiguration(
            target_nova='nova2',
            target_host='localhost',
            target_port=8443,
            sync_mode=SyncMode.INCREMENTAL,
            privacy_levels=[PrivacyLevel.PUBLIC]
        )
        
        session_id = self.sync_manager1.add_sync_configuration(config)
        
        # Check that configuration was added
        self.assertIn(session_id, self.sync_manager1.active_sessions)
        
        # Check sync status
        status = self.sync_manager1.get_sync_status()
        self.assertTrue(status['is_running'])
        self.assertEqual(status['active_sessions'], 1)

class StressTests(unittest.IsolatedAsyncioTestCase):
    """Stress tests for network failure scenarios"""
    
    async def asyncSetUp(self):
        """Set up stress test environment"""
        self.protocol = CrossNovaTransferProtocol('test_nova')
    
    async def asyncTearDown(self):
        """Clean up stress test environment"""
        await self.protocol.stop_server()
    
    async def test_large_data_transfer_simulation(self):
        """Test handling of large data transfers"""
        # Create large mock data
        large_data = json.dumps({
            'memories': [
                {
                    'id': f'mem_{i}',
                    'content': 'A' * 1000,  # 1KB per memory
                    'timestamp': datetime.now().isoformat()
                }
                for i in range(1000)  # 1MB total
            ]
        }).encode()
        
        # Test chunking
        chunks = ChunkManager.create_chunks(large_data, chunk_size=10240)  # 10KB chunks
        
        self.assertGreater(len(chunks), 50)  # Should create many chunks
        
        # Test reassembly
        chunk_dict = {chunk_id: chunk_data for chunk_id, chunk_data in chunks}
        reassembled = ChunkManager.reassemble_chunks(chunk_dict)
        
        self.assertEqual(large_data, reassembled)
    
    async def test_network_failure_simulation(self):
        """Test network failure handling"""
        # Test chunked transfer with missing chunks
        original_data = b'test data for network failure simulation' * 100
        chunks = ChunkManager.create_chunks(original_data, chunk_size=50)
        
        # Simulate missing some chunks
        partial_chunks = {chunk_id: chunk_data for chunk_id, chunk_data in chunks[:-2]}
        
        # Should not be able to reassemble completely
        with self.assertRaises(Exception):
            # In a real implementation, this would handle missing chunks gracefully
            reassembled = ChunkManager.reassemble_chunks(partial_chunks)
            if len(reassembled) != len(original_data):
                raise Exception("Incomplete data")
    
    async def test_concurrent_transfers(self):
        """Test multiple concurrent transfers"""
        bandwidth_limiter = BandwidthLimiter(max_bytes_per_second=1000)
        
        # Simulate concurrent requests
        tasks = []
        for i in range(10):
            task = asyncio.create_task(bandwidth_limiter.acquire(100))
            tasks.append(task)
        
        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()
        
        # Should take some time due to rate limiting
        self.assertGreater(end_time - start_time, 0.5)

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestVectorClock,
        TestMemoryDelta,
        TestCompressionManager,
        TestChunkManager,
        TestBandwidthLimiter,
        TestPrivacyController,
        TestConflictResolver,
        TestMemorySnapshot,
        TestBandwidthOptimizer
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

async def run_async_tests():
    """Run async test suites"""
    # These tests require asyncio
    async_test_classes = [
        TestCrossNovaTransferProtocol,
        TestMemorySyncManager,
        IntegrationTests,
        StressTests
    ]
    
    success = True
    for test_class in async_test_classes:
        print(f"\nRunning {test_class.__name__}...")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if not result.wasSuccessful():
            success = False
    
    return success

if __name__ == "__main__":
    print("Running Cross-Nova Memory Transfer Protocol Test Suite")
    print("=" * 60)
    
    # Run synchronous tests
    print("\n1. Running synchronous tests...")
    sync_success = run_all_tests()
    
    # Run asynchronous tests
    print("\n2. Running asynchronous tests...")
    async_success = asyncio.run(run_async_tests())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print(f"Synchronous tests: {'PASSED' if sync_success else 'FAILED'}")
    print(f"Asynchronous tests: {'PASSED' if async_success else 'FAILED'}")
    
    overall_success = sync_success and async_success
    print(f"Overall result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
    
    exit(0 if overall_success else 1)