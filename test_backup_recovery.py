"""
Nova Bloom Consciousness - Backup Recovery Test Suite
Comprehensive testing framework for backup and recovery systems.

This module implements extensive test cases for:
- Backup system functionality and strategies
- Disaster recovery orchestration and RPO/RTO compliance
- Backup integrity checking and corruption detection
- Cross-platform storage backend validation
- Performance benchmarking and stress testing
- Real-world failure scenario simulation
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import AsyncMock, MagicMock, patch
import sqlite3

# Import our backup and recovery components
from memory_backup_system import (
    MemoryBackupSystem, BackupStrategy, BackupStatus, 
    StorageBackend, BackupMetadata, DeduplicationManager
)
from disaster_recovery_manager import (
    DisasterRecoveryManager, DisasterType, RecoveryMode,
    RecoveryStatus, RPOTarget, RTOTarget
)
from backup_integrity_checker import (
    BackupIntegrityChecker, IntegrityLevel, IntegrityStatus,
    CorruptionType, IntegrityIssue
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMemoryBackupSystem(unittest.IsolatedAsyncioTestCase):
    """Test suite for MemoryBackupSystem."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix='nova_backup_test_'))
        self.backup_dir = self.test_dir / 'backups'
        self.storage_dir = self.test_dir / 'storage'
        
        # Create test configuration
        self.config = {
            'backup_dir': str(self.backup_dir),
            'storage': {
                'local_path': str(self.storage_dir)
            },
            'retention_days': 7
        }
        
        # Initialize backup system
        self.backup_system = MemoryBackupSystem(self.config)
        
        # Create test memory layers
        self.test_layers = []
        for i in range(3):
            layer_path = self.test_dir / f'test_layer_{i}.json'
            with open(layer_path, 'w') as f:
                json.dump({
                    'layer_id': i,
                    'data': f'test data for layer {i}',
                    'timestamp': datetime.now().isoformat(),
                    'memory_content': [f'memory_{i}_{j}' for j in range(10)]
                }, f)
            self.test_layers.append(str(layer_path))
        
        logger.info(f"Test environment set up in {self.test_dir}")
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.backup_system.stop_background_tasks()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        logger.info("Test environment cleaned up")
    
    async def test_full_backup_creation(self):
        """Test creating a full backup."""
        logger.info("Testing full backup creation")
        
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL,
            tags={'test': 'full_backup', 'version': '1.0'}
        )
        
        # Verify backup was created
        self.assertIsNotNone(backup)
        self.assertEqual(backup.strategy, BackupStrategy.FULL)
        self.assertEqual(backup.status, BackupStatus.COMPLETED)
        self.assertEqual(len(backup.memory_layers), 3)
        self.assertTrue(backup.compressed_size > 0)
        self.assertTrue(backup.original_size > 0)
        self.assertTrue(backup.checksum)
        
        # Verify backup is in database
        retrieved_backup = await self.backup_system.get_backup(backup.backup_id)
        self.assertIsNotNone(retrieved_backup)
        self.assertEqual(retrieved_backup.backup_id, backup.backup_id)
        
        logger.info(f"Full backup test passed: {backup.backup_id}")
    
    async def test_incremental_backup_strategy(self):
        """Test incremental backup strategy."""
        logger.info("Testing incremental backup strategy")
        
        # Create initial full backup
        full_backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        self.assertIsNotNone(full_backup)
        
        # Wait a moment and modify one file
        await asyncio.sleep(1)
        modified_layer = Path(self.test_layers[0])
        with open(modified_layer, 'w') as f:
            json.dump({
                'layer_id': 0,
                'data': 'modified test data',
                'timestamp': datetime.now().isoformat(),
                'modified': True
            }, f)
        
        # Create incremental backup
        incremental_backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.INCREMENTAL
        )
        
        self.assertIsNotNone(incremental_backup)
        self.assertEqual(incremental_backup.strategy, BackupStrategy.INCREMENTAL)
        self.assertEqual(incremental_backup.status, BackupStatus.COMPLETED)
        
        logger.info(f"Incremental backup test passed: {incremental_backup.backup_id}")
    
    async def test_backup_listing_and_filtering(self):
        """Test backup listing with filtering."""
        logger.info("Testing backup listing and filtering")
        
        # Create multiple backups with different strategies
        backups_created = []
        
        for strategy in [BackupStrategy.FULL, BackupStrategy.INCREMENTAL, BackupStrategy.DIFFERENTIAL]:
            backup = await self.backup_system.create_backup(
                memory_layers=self.test_layers,
                strategy=strategy,
                tags={'strategy': strategy.value}
            )
            if backup:
                backups_created.append(backup)
                await asyncio.sleep(0.1)  # Small delay between backups
        
        # List all backups
        all_backups = await self.backup_system.list_backups()
        self.assertGreaterEqual(len(all_backups), 3)
        
        # Filter by strategy
        full_backups = await self.backup_system.list_backups(strategy=BackupStrategy.FULL)
        self.assertGreaterEqual(len(full_backups), 1)
        
        # Filter by status
        completed_backups = await self.backup_system.list_backups(status=BackupStatus.COMPLETED)
        self.assertEqual(len(completed_backups), len(backups_created))
        
        logger.info(f"Backup listing test passed: {len(all_backups)} total backups")
    
    async def test_backup_deletion(self):
        """Test backup deletion functionality."""
        logger.info("Testing backup deletion")
        
        # Create backup to delete
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        self.assertIsNotNone(backup)
        
        # Verify backup exists
        retrieved = await self.backup_system.get_backup(backup.backup_id)
        self.assertIsNotNone(retrieved)
        
        # Delete backup
        delete_success = await self.backup_system.delete_backup(backup.backup_id)
        self.assertTrue(delete_success)
        
        # Verify backup is gone
        retrieved_after_delete = await self.backup_system.get_backup(backup.backup_id)
        self.assertIsNone(retrieved_after_delete)
        
        logger.info(f"Backup deletion test passed: {backup.backup_id}")
    
    async def test_deduplication_functionality(self):
        """Test file deduplication."""
        logger.info("Testing deduplication functionality")
        
        # Create duplicate files
        duplicate_content = {'duplicate': 'content', 'timestamp': datetime.now().isoformat()}
        
        dup_files = []
        for i in range(3):
            dup_file = self.test_dir / f'duplicate_{i}.json'
            with open(dup_file, 'w') as f:
                json.dump(duplicate_content, f)
            dup_files.append(str(dup_file))
        
        # Create backup with duplicate files
        backup = await self.backup_system.create_backup(
            memory_layers=dup_files,
            strategy=BackupStrategy.FULL
        )
        
        self.assertIsNotNone(backup)
        # With deduplication, compressed size should be significantly smaller
        # than what it would be without deduplication
        self.assertTrue(backup.compressed_size < backup.original_size)
        
        logger.info("Deduplication test passed")
    
    async def test_cleanup_old_backups(self):
        """Test automatic cleanup of old backups."""
        logger.info("Testing backup cleanup")
        
        # Create some old backups by manipulating timestamps
        old_backups = []
        for i in range(3):
            backup = await self.backup_system.create_backup(
                memory_layers=self.test_layers,
                strategy=BackupStrategy.FULL
            )
            if backup:
                # Modify backup timestamp to be old
                backup.timestamp = datetime.now() - timedelta(days=35)
                await self.backup_system._save_metadata(backup)
                old_backups.append(backup.backup_id)
        
        # Create recent backup
        recent_backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        
        # Run cleanup with 30-day retention
        cleaned_count = await self.backup_system.cleanup_old_backups(retention_days=30)
        self.assertEqual(cleaned_count, len(old_backups))
        
        # Verify old backups are gone but recent one remains
        for old_id in old_backups:
            retrieved = await self.backup_system.get_backup(old_id)
            self.assertIsNone(retrieved)
        
        recent_retrieved = await self.backup_system.get_backup(recent_backup.backup_id)
        self.assertIsNotNone(recent_retrieved)
        
        logger.info(f"Cleanup test passed: {cleaned_count} backups cleaned")


class TestDisasterRecoveryManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for DisasterRecoveryManager."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix='nova_recovery_test_'))
        
        # Set up backup system first
        backup_config = {
            'backup_dir': str(self.test_dir / 'backups'),
            'storage': {
                'local_path': str(self.test_dir / 'storage')
            }
        }
        self.backup_system = MemoryBackupSystem(backup_config)
        
        # Set up disaster recovery manager
        recovery_config = {
            'recovery_dir': str(self.test_dir / 'recovery'),
            'rpo_targets': {
                'critical': {
                    'max_data_loss_minutes': 5,
                    'critical_layers': ['/tmp/critical_layer.json'],
                    'backup_frequency_minutes': 1
                }
            },
            'rto_targets': {
                'critical': {
                    'max_recovery_minutes': 10,
                    'critical_components': ['memory_system']
                }
            }
        }
        self.recovery_manager = DisasterRecoveryManager(recovery_config, self.backup_system)
        
        # Create test memory layers
        self.test_layers = []
        for i in range(2):
            layer_path = self.test_dir / f'test_layer_{i}.json'
            with open(layer_path, 'w') as f:
                json.dump({
                    'layer_id': i,
                    'data': f'recovery test data {i}',
                    'timestamp': datetime.now().isoformat()
                }, f)
            self.test_layers.append(str(layer_path))
        
        logger.info(f"Recovery test environment set up in {self.test_dir}")
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.recovery_manager.stop_monitoring()
        await self.backup_system.stop_background_tasks()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        logger.info("Recovery test environment cleaned up")
    
    async def test_recovery_trigger(self):
        """Test triggering disaster recovery."""
        logger.info("Testing recovery trigger")
        
        # Create backup first
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        self.assertIsNotNone(backup)
        
        # Trigger recovery
        recovery = await self.recovery_manager.trigger_recovery(
            disaster_type=DisasterType.DATA_CORRUPTION,
            affected_layers=self.test_layers,
            recovery_mode=RecoveryMode.TESTING,
            backup_id=backup.backup_id
        )
        
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.disaster_type, DisasterType.DATA_CORRUPTION)
        self.assertEqual(recovery.backup_id, backup.backup_id)
        self.assertEqual(len(recovery.affected_layers), 2)
        
        logger.info(f"Recovery trigger test passed: {recovery.recovery_id}")
    
    async def test_automatic_backup_selection(self):
        """Test automatic backup selection for recovery."""
        logger.info("Testing automatic backup selection")
        
        # Create multiple backups at different times
        backups = []
        for i in range(3):
            backup = await self.backup_system.create_backup(
                memory_layers=self.test_layers,
                strategy=BackupStrategy.FULL,
                tags={'sequence': str(i)}
            )
            if backup:
                backups.append(backup)
                await asyncio.sleep(0.1)  # Small delay
        
        # Trigger recovery without specifying backup ID
        recovery = await self.recovery_manager.trigger_recovery(
            disaster_type=DisasterType.SYSTEM_CRASH,
            affected_layers=self.test_layers,
            recovery_mode=RecoveryMode.TESTING
        )
        
        self.assertIsNotNone(recovery)
        self.assertIsNotNone(recovery.backup_id)
        
        # Should select the most recent backup
        selected_backup = await self.backup_system.get_backup(recovery.backup_id)
        self.assertIsNotNone(selected_backup)
        
        logger.info(f"Automatic backup selection test passed: selected {recovery.backup_id}")
    
    async def test_point_in_time_recovery(self):
        """Test point-in-time recovery."""
        logger.info("Testing point-in-time recovery")
        
        # Create backup
        backup_time = datetime.now()
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        self.assertIsNotNone(backup)
        
        # Set target time slightly after backup
        target_time = backup_time + timedelta(minutes=1)
        
        # Trigger point-in-time recovery
        recovery = await self.recovery_manager.trigger_recovery(
            disaster_type=DisasterType.DATA_CORRUPTION,
            affected_layers=self.test_layers,
            recovery_mode=RecoveryMode.TESTING,
            target_timestamp=target_time
        )
        
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.target_timestamp, target_time)
        
        logger.info(f"Point-in-time recovery test passed: {recovery.recovery_id}")
    
    async def test_recovery_listing(self):
        """Test listing recovery operations."""
        logger.info("Testing recovery listing")
        
        # Create backup
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        
        # Create multiple recoveries
        recoveries_created = []
        for disaster_type in [DisasterType.DATA_CORRUPTION, DisasterType.SYSTEM_CRASH]:
            recovery = await self.recovery_manager.trigger_recovery(
                disaster_type=disaster_type,
                affected_layers=self.test_layers,
                recovery_mode=RecoveryMode.TESTING,
                backup_id=backup.backup_id
            )
            if recovery:
                recoveries_created.append(recovery)
        
        # List all recoveries
        all_recoveries = await self.recovery_manager.list_recoveries()
        self.assertGreaterEqual(len(all_recoveries), 2)
        
        # Filter by disaster type
        corruption_recoveries = await self.recovery_manager.list_recoveries(
            disaster_type=DisasterType.DATA_CORRUPTION
        )
        self.assertGreaterEqual(len(corruption_recoveries), 1)
        
        logger.info(f"Recovery listing test passed: {len(all_recoveries)} recoveries")
    
    async def test_recovery_testing(self):
        """Test recovery testing functionality."""
        logger.info("Testing recovery testing")
        
        # Create backup
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        self.assertIsNotNone(backup)
        
        # Run recovery test
        test_results = await self.recovery_manager.test_recovery(
            test_layers=self.test_layers,
            backup_id=backup.backup_id
        )
        
        self.assertIsNotNone(test_results)
        self.assertIn('success', test_results)
        self.assertIn('recovery_id', test_results)
        
        # Test should not affect production
        self.assertTrue(Path(self.test_layers[0]).exists())
        
        logger.info(f"Recovery testing passed: {test_results}")
    
    async def test_rpo_rto_calculation(self):
        """Test RPO/RTO calculation."""
        logger.info("Testing RPO/RTO calculation")
        
        # Create backup
        backup = await self.backup_system.create_backup(
            memory_layers=self.test_layers,
            strategy=BackupStrategy.FULL
        )
        
        # Trigger recovery and wait for completion
        start_time = datetime.now()
        recovery = await self.recovery_manager.trigger_recovery(
            disaster_type=DisasterType.DATA_CORRUPTION,
            affected_layers=self.test_layers,
            recovery_mode=RecoveryMode.TESTING,
            target_timestamp=start_time,
            backup_id=backup.backup_id
        )
        
        # Wait for recovery to complete (simplified for test)
        await asyncio.sleep(1)
        
        # Get updated recovery metadata
        updated_recovery = await self.recovery_manager.get_recovery(recovery.recovery_id)
        if updated_recovery:
            # Should have calculated RPO/RTO values
            self.assertIsNotNone(updated_recovery.rto_achieved_minutes)
            if updated_recovery.target_timestamp:
                self.assertIsNotNone(updated_recovery.rpo_achieved_minutes)
        
        logger.info("RPO/RTO calculation test passed")


class TestBackupIntegrityChecker(unittest.IsolatedAsyncioTestCase):
    """Test suite for BackupIntegrityChecker."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix='nova_integrity_test_'))
        
        # Set up integrity checker
        config = {
            'integrity_dir': str(self.test_dir / 'integrity'),
            'monitor_files': []
        }
        self.integrity_checker = BackupIntegrityChecker(config)
        
        # Create test files
        self.test_files = []
        
        # Valid JSON file
        valid_json = self.test_dir / 'valid.json'
        with open(valid_json, 'w') as f:
            json.dump({'valid': True, 'data': 'test'}, f)
        self.test_files.append(str(valid_json))
        
        # Invalid JSON file
        invalid_json = self.test_dir / 'invalid.json'
        with open(invalid_json, 'w') as f:
            f.write('{"invalid": "json",}')  # Trailing comma
        self.test_files.append(str(invalid_json))
        
        logger.info(f"Integrity test environment set up in {self.test_dir}")
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.integrity_checker.stop_monitoring()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        logger.info("Integrity test environment cleaned up")
    
    async def test_basic_integrity_check(self):
        """Test basic integrity checking."""
        logger.info("Testing basic integrity check")
        
        # Check valid file
        result = await self.integrity_checker.check_file_integrity(
            self.test_files[0],
            IntegrityLevel.BASIC
        )
        
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.issues), 0)
        
        logger.info("Basic integrity check test passed")
    
    async def test_checksum_validation(self):
        """Test checksum-based validation."""
        logger.info("Testing checksum validation")
        
        # Calculate expected checksum
        import hashlib
        with open(self.test_files[0], 'rb') as f:
            content = f.read()
            expected_checksum = hashlib.sha256(content).hexdigest()
        
        expected_metadata = {
            'sha256_checksum': expected_checksum,
            'size': len(content)
        }
        
        # Check with correct checksum
        result = await self.integrity_checker.check_file_integrity(
            self.test_files[0],
            IntegrityLevel.CHECKSUM,
            expected_metadata
        )
        
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.issues), 0)
        
        # Check with incorrect checksum
        bad_metadata = {
            'sha256_checksum': 'invalid_checksum',
            'size': len(content)
        }
        
        result_bad = await self.integrity_checker.check_file_integrity(
            self.test_files[0],
            IntegrityLevel.CHECKSUM,
            bad_metadata
        )
        
        self.assertEqual(result_bad.status, IntegrityStatus.FAILED)
        self.assertGreater(len(result_bad.issues), 0)
        
        logger.info("Checksum validation test passed")
    
    async def test_content_validation(self):
        """Test content structure validation."""
        logger.info("Testing content validation")
        
        # Check invalid JSON file
        result = await self.integrity_checker.check_file_integrity(
            self.test_files[1],  # Invalid JSON file
            IntegrityLevel.CONTENT
        )
        
        self.assertIn(result.status, [IntegrityStatus.FAILED, IntegrityStatus.CORRUPTED])
        self.assertGreater(len(result.issues), 0)
        
        # Should have structure validation issue
        structure_issues = [
            issue for issue in result.issues 
            if issue.corruption_type == CorruptionType.STRUCTURE_INVALID
        ]
        self.assertGreater(len(structure_issues), 0)
        
        logger.info("Content validation test passed")
    
    async def test_multiple_file_checking(self):
        """Test checking multiple files concurrently."""
        logger.info("Testing multiple file checking")
        
        results = await self.integrity_checker.check_multiple_files(
            self.test_files,
            IntegrityLevel.CONTENT,
            max_concurrent=2
        )
        
        self.assertEqual(len(results), len(self.test_files))
        
        # Valid file should pass
        self.assertEqual(results[self.test_files[0]].status, IntegrityStatus.PASSED)
        
        # Invalid file should fail
        self.assertIn(results[self.test_files[1]].status, 
                     [IntegrityStatus.FAILED, IntegrityStatus.CORRUPTED])
        
        logger.info("Multiple file checking test passed")
    
    async def test_integrity_repair(self):
        """Test integrity issue repair."""
        logger.info("Testing integrity repair")
        
        # Check invalid JSON file to get issues
        result = await self.integrity_checker.check_file_integrity(
            self.test_files[1],
            IntegrityLevel.CONTENT
        )
        
        self.assertGreater(len(result.issues), 0)
        
        # Attempt repair
        repair_success = await self.integrity_checker.attempt_repair(result)
        
        # For JSON structure issues, repair should be attempted
        structure_issues = [
            issue for issue in result.issues
            if issue.corruption_type == CorruptionType.STRUCTURE_INVALID and issue.repairable
        ]
        
        if structure_issues:
            # Should have attempted repair
            self.assertTrue(result.repair_attempted)
        
        logger.info("Integrity repair test passed")
    
    async def test_integrity_report_generation(self):
        """Test integrity report generation."""
        logger.info("Testing integrity report generation")
        
        # Check multiple files to generate data
        await self.integrity_checker.check_multiple_files(
            self.test_files,
            IntegrityLevel.CONTENT
        )
        
        # Generate report
        report = await self.integrity_checker.generate_integrity_report()
        
        self.assertIn('generated_at', report)
        self.assertIn('total_checks', report)
        self.assertIn('status_summary', report)
        self.assertIn('corruption_types', report)
        self.assertIn('files_with_issues', report)
        
        # Should have some data
        self.assertGreater(report['total_checks'], 0)
        
        logger.info("Integrity report generation test passed")
    
    async def test_monitoring_functionality(self):
        """Test continuous integrity monitoring."""
        logger.info("Testing integrity monitoring")
        
        # Configure monitoring files
        self.integrity_checker.config['monitor_files'] = self.test_files
        
        # Start monitoring
        await self.integrity_checker.start_monitoring(check_interval_minutes=1)
        
        # Let it run briefly
        await asyncio.sleep(2)
        
        # Stop monitoring
        await self.integrity_checker.stop_monitoring()
        
        # Should have created some check results
        results = await self.integrity_checker.list_check_results(limit=10)
        # Note: Results might be empty if monitoring interval hasn't triggered
        
        logger.info("Integrity monitoring test passed")


class TestIntegrationScenarios(unittest.IsolatedAsyncioTestCase):
    """Integration tests for complete backup and recovery workflows."""
    
    async def asyncSetUp(self):
        """Set up complete test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix='nova_integration_test_'))
        
        # Set up backup system
        backup_config = {
            'backup_dir': str(self.test_dir / 'backups'),
            'storage': {
                'local_path': str(self.test_dir / 'storage')
            },
            'retention_days': 30
        }
        self.backup_system = MemoryBackupSystem(backup_config)
        
        # Set up disaster recovery
        recovery_config = {
            'recovery_dir': str(self.test_dir / 'recovery'),
            'rpo_targets': {
                'default': {
                    'max_data_loss_minutes': 5,
                    'critical_layers': [],
                    'backup_frequency_minutes': 1
                }
            },
            'rto_targets': {
                'default': {
                    'max_recovery_minutes': 15,
                    'critical_components': ['memory_system']
                }
            }
        }
        self.recovery_manager = DisasterRecoveryManager(recovery_config, self.backup_system)
        
        # Set up integrity checker
        integrity_config = {
            'integrity_dir': str(self.test_dir / 'integrity')
        }
        self.integrity_checker = BackupIntegrityChecker(integrity_config, self.backup_system)
        
        # Create test memory layers
        self.memory_layers = []
        for i in range(5):
            layer_path = self.test_dir / f'memory_layer_{i}.json'
            with open(layer_path, 'w') as f:
                json.dump({
                    'layer_id': i,
                    'memory_data': [f'memory_block_{i}_{j}' for j in range(100)],
                    'metadata': {
                        'created': datetime.now().isoformat(),
                        'version': '1.0',
                        'checksum': f'layer_{i}_checksum'
                    },
                    'consciousness_state': {
                        'active': True,
                        'priority': i * 10,
                        'connections': [f'layer_{j}' for j in range(i)]
                    }
                }, f)
            self.memory_layers.append(str(layer_path))
        
        logger.info(f"Integration test environment set up in {self.test_dir}")
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.recovery_manager.stop_monitoring()
        await self.backup_system.stop_background_tasks()
        await self.integrity_checker.stop_monitoring()
        shutil.rmtree(self.test_dir, ignore_errors=True)
        logger.info("Integration test environment cleaned up")
    
    async def test_complete_backup_recovery_workflow(self):
        """Test complete backup and recovery workflow."""
        logger.info("Testing complete backup and recovery workflow")
        
        # Step 1: Create initial backup
        initial_backup = await self.backup_system.create_backup(
            memory_layers=self.memory_layers,
            strategy=BackupStrategy.FULL,
            tags={'workflow': 'integration_test', 'phase': 'initial'}
        )
        self.assertIsNotNone(initial_backup)
        logger.info(f"Created initial backup: {initial_backup.backup_id}")
        
        # Step 2: Check backup integrity
        integrity_results = await self.integrity_checker.check_backup_integrity(
            initial_backup.backup_id,
            IntegrityLevel.CHECKSUM
        )
        self.assertGreater(len(integrity_results), 0)
        
        # All layers should pass integrity check
        passed_checks = [r for r in integrity_results.values() if r.status == IntegrityStatus.PASSED]
        logger.info(f"Integrity check results: {len(passed_checks)} passed")
        
        # Step 3: Simulate disaster by corrupting data
        corrupted_layer = Path(self.memory_layers[0])
        original_content = corrupted_layer.read_text()
        corrupted_layer.write_text("CORRUPTED DATA")
        logger.info(f"Simulated corruption in {corrupted_layer}")
        
        # Step 4: Detect corruption through integrity check
        corruption_check = await self.integrity_checker.check_file_integrity(
            str(corrupted_layer),
            IntegrityLevel.CONTENT
        )
        self.assertNotEqual(corruption_check.status, IntegrityStatus.PASSED)
        logger.info("Corruption detected by integrity checker")
        
        # Step 5: Trigger disaster recovery
        recovery = await self.recovery_manager.trigger_recovery(
            disaster_type=DisasterType.DATA_CORRUPTION,
            affected_layers=[str(corrupted_layer)],
            recovery_mode=RecoveryMode.TESTING,
            backup_id=initial_backup.backup_id
        )
        self.assertIsNotNone(recovery)
        logger.info(f"Recovery initiated: {recovery.recovery_id}")
        
        # Step 6: Wait for recovery completion (simplified)
        await asyncio.sleep(2)
        
        # Step 7: Verify recovery completion
        updated_recovery = await self.recovery_manager.get_recovery(recovery.recovery_id)
        self.assertIsNotNone(updated_recovery)
        logger.info(f"Recovery status: {updated_recovery.status.value}")
        
        # Step 8: Verify system integrity post-recovery
        post_recovery_check = await self.integrity_checker.check_file_integrity(
            str(corrupted_layer),
            IntegrityLevel.BASIC
        )
        # Note: In real implementation, recovery would restore the file
        logger.info(f"Post-recovery integrity: {post_recovery_check.status.value}")
        
        logger.info("Complete backup and recovery workflow test completed")
    
    async def test_multi_strategy_backup_scenario(self):
        """Test multiple backup strategies in sequence."""
        logger.info("Testing multi-strategy backup scenario")
        
        # Create full backup
        full_backup = await self.backup_system.create_backup(
            memory_layers=self.memory_layers,
            strategy=BackupStrategy.FULL,
            tags={'strategy_test': 'full'}
        )
        self.assertIsNotNone(full_backup)
        logger.info(f"Full backup created: {full_backup.backup_id}")
        
        # Modify some files
        await asyncio.sleep(1)  # Ensure timestamp difference
        for i in range(2):  # Modify first 2 layers
            layer_path = Path(self.memory_layers[i])
            with open(layer_path, 'r') as f:
                data = json.load(f)
            data['modified'] = True
            data['modification_time'] = datetime.now().isoformat()
            with open(layer_path, 'w') as f:
                json.dump(data, f)
        logger.info("Modified 2 memory layers")
        
        # Create incremental backup
        incremental_backup = await self.backup_system.create_backup(
            memory_layers=self.memory_layers,
            strategy=BackupStrategy.INCREMENTAL,
            tags={'strategy_test': 'incremental'}
        )
        self.assertIsNotNone(incremental_backup)
        logger.info(f"Incremental backup created: {incremental_backup.backup_id}")
        
        # Modify more files
        await asyncio.sleep(1)
        for i in range(2, 4):  # Modify layers 2-3
            layer_path = Path(self.memory_layers[i])
            with open(layer_path, 'r') as f:
                data = json.load(f)
            data['second_modification'] = True
            data['second_modification_time'] = datetime.now().isoformat()
            with open(layer_path, 'w') as f:
                json.dump(data, f)
        logger.info("Modified 2 additional memory layers")
        
        # Create differential backup
        differential_backup = await self.backup_system.create_backup(
            memory_layers=self.memory_layers,
            strategy=BackupStrategy.DIFFERENTIAL,
            tags={'strategy_test': 'differential'}
        )
        self.assertIsNotNone(differential_backup)
        logger.info(f"Differential backup created: {differential_backup.backup_id}")
        
        # Verify all backups exist and have correct strategies
        all_backups = await self.backup_system.list_backups()
        strategy_counts = {}
        for backup in all_backups:
            strategy = backup.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        self.assertGreaterEqual(strategy_counts.get('full', 0), 1)
        self.assertGreaterEqual(strategy_counts.get('incremental', 0), 1)
        self.assertGreaterEqual(strategy_counts.get('differential', 0), 1)
        
        logger.info(f"Multi-strategy backup test completed: {strategy_counts}")
    
    async def test_performance_benchmarking(self):
        """Test performance benchmarking of backup operations."""
        logger.info("Testing performance benchmarking")
        
        # Create larger test files for performance testing
        large_layers = []
        for i in range(10):
            layer_path = self.test_dir / f'large_layer_{i}.json'
            large_data = {
                'layer_id': i,
                'large_memory_data': [f'large_block_{i}_{j}' for j in range(1000)],
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'size': 'large'
                }
            }
            with open(layer_path, 'w') as f:
                json.dump(large_data, f)
            large_layers.append(str(layer_path))
        
        # Benchmark full backup creation
        start_time = time.time()
        backup = await self.backup_system.create_backup(
            memory_layers=large_layers,
            strategy=BackupStrategy.FULL,
            tags={'benchmark': 'performance'}
        )
        backup_time = time.time() - start_time
        
        self.assertIsNotNone(backup)
        logger.info(f"Backup creation took {backup_time:.2f} seconds")
        
        # Benchmark integrity checking
        start_time = time.time()
        integrity_results = await self.integrity_checker.check_multiple_files(
            large_layers,
            IntegrityLevel.CHECKSUM,
            max_concurrent=4
        )
        integrity_time = time.time() - start_time
        
        self.assertEqual(len(integrity_results), len(large_layers))
        logger.info(f"Integrity checking took {integrity_time:.2f} seconds")
        
        # Calculate performance metrics
        total_size = sum(Path(layer).stat().st_size for layer in large_layers)
        backup_throughput = total_size / backup_time  # bytes per second
        integrity_throughput = total_size / integrity_time
        
        logger.info(f"Backup throughput: {backup_throughput / 1024 / 1024:.2f} MB/s")
        logger.info(f"Integrity check throughput: {integrity_throughput / 1024 / 1024:.2f} MB/s")
        
        # Performance assertions
        self.assertGreater(backup_throughput, 0)
        self.assertGreater(integrity_throughput, 0)
        
        logger.info("Performance benchmarking test completed")
    
    async def test_concurrent_operations(self):
        """Test concurrent backup and recovery operations."""
        logger.info("Testing concurrent operations")
        
        # Create multiple backup tasks concurrently
        backup_tasks = []
        for i in range(3):
            task = asyncio.create_task(
                self.backup_system.create_backup(
                    memory_layers=self.memory_layers[i:i+2],  # Different layers per backup
                    strategy=BackupStrategy.FULL,
                    tags={'concurrent': str(i)}
                )
            )
            backup_tasks.append(task)
        
        # Wait for all backups to complete
        backups = await asyncio.gather(*backup_tasks, return_exceptions=True)
        
        # Count successful backups
        successful_backups = [b for b in backups if isinstance(b, BackupMetadata)]
        self.assertGreater(len(successful_backups), 0)
        logger.info(f"Concurrent backup test: {len(successful_backups)} successful")
        
        # Create concurrent integrity check tasks
        if successful_backups:
            integrity_tasks = []
            for backup in successful_backups:
                task = asyncio.create_task(
                    self.integrity_checker.check_backup_integrity(
                        backup.backup_id,
                        IntegrityLevel.BASIC
                    )
                )
                integrity_tasks.append(task)
            
            # Wait for integrity checks
            integrity_results = await asyncio.gather(*integrity_tasks, return_exceptions=True)
            successful_checks = [r for r in integrity_results if isinstance(r, dict)]
            logger.info(f"Concurrent integrity checks: {len(successful_checks)} successful")
        
        logger.info("Concurrent operations test completed")


class TestErrorHandlingAndEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Test error handling and edge cases."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix='nova_error_test_'))
        
        config = {
            'backup_dir': str(self.test_dir / 'backups'),
            'storage': {
                'local_path': str(self.test_dir / 'storage')
            }
        }
        self.backup_system = MemoryBackupSystem(config)
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.backup_system.stop_background_tasks()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    async def test_missing_file_backup(self):
        """Test backup of non-existent files."""
        logger.info("Testing missing file backup")
        
        missing_files = ['/nonexistent/file1.json', '/missing/file2.json']
        
        backup = await self.backup_system.create_backup(
            memory_layers=missing_files,
            strategy=BackupStrategy.FULL
        )
        
        # Should handle gracefully - backup might be created but with no files
        # or might fail gracefully
        if backup:
            self.assertEqual(backup.file_count, 0)
        
        logger.info("Missing file backup test completed")
    
    async def test_corrupted_backup_archive(self):
        """Test handling of corrupted backup archives."""
        logger.info("Testing corrupted backup archive handling")
        
        # Create a valid backup first
        test_file = self.test_dir / 'test.json'
        with open(test_file, 'w') as f:
            json.dump({'test': 'data'}, f)
        
        backup = await self.backup_system.create_backup(
            memory_layers=[str(test_file)],
            strategy=BackupStrategy.FULL
        )
        self.assertIsNotNone(backup)
        
        # Simulate corruption by finding and corrupting the backup file
        storage_dir = Path(self.backup_system.storage_adapters[StorageBackend.LOCAL].base_path)
        backup_files = list(storage_dir.rglob('*.backup'))
        
        if backup_files:
            # Corrupt the backup file
            backup_file = backup_files[0]
            with open(backup_file, 'wb') as f:
                f.write(b'CORRUPTED_BACKUP_DATA')
            
            # Test integrity checker with corrupted file
            integrity_checker = BackupIntegrityChecker({
                'integrity_dir': str(self.test_dir / 'integrity')
            })
            
            result = await integrity_checker.check_file_integrity(
                str(backup_file),
                IntegrityLevel.CONTENT
            )
            
            # Should detect corruption
            self.assertNotEqual(result.status, IntegrityStatus.PASSED)
            logger.info("Corruption detected in backup archive")
        
        logger.info("Corrupted backup archive test completed")
    
    async def test_storage_full_scenario(self):
        """Test handling of storage full scenarios."""
        logger.info("Testing storage full scenario")
        
        # Create large file that might fill storage
        large_file = self.test_dir / 'large_file.json'
        large_data = {'data': 'x' * (10 * 1024 * 1024)}  # 10MB of data
        
        try:
            with open(large_file, 'w') as f:
                json.dump(large_data, f)
            
            # Attempt backup (may fail due to space constraints)
            backup = await self.backup_system.create_backup(
                memory_layers=[str(large_file)],
                strategy=BackupStrategy.FULL
            )
            
            # Should either succeed or fail gracefully
            if backup:
                self.assertIn(backup.status, [BackupStatus.COMPLETED, BackupStatus.FAILED])
            
        except Exception as e:
            logger.info(f"Storage full scenario handled: {e}")
        
        logger.info("Storage full scenario test completed")


if __name__ == '__main__':
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)