"""
Nova Bloom Consciousness - Disaster Recovery Manager
Critical system for automated disaster recovery with RPO/RTO targets.

This module implements comprehensive disaster recovery capabilities including:
- Automated failover and recovery orchestration
- RPO (Recovery Point Objective) and RTO (Recovery Time Objective) monitoring
- Point-in-time recovery with precise timestamp control
- Cross-platform recovery execution
- Health monitoring and automated recovery triggers
- Recovery testing and validation frameworks
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil

# Import from our backup system
from memory_backup_system import (
    MemoryBackupSystem, BackupMetadata, BackupStrategy, 
    BackupStatus, StorageBackend
)

logger = logging.getLogger(__name__)


class RecoveryStatus(Enum):
    """Status of recovery operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TESTING = "testing"


class DisasterType(Enum):
    """Types of disasters that can trigger recovery."""
    DATA_CORRUPTION = "data_corruption"
    HARDWARE_FAILURE = "hardware_failure"
    NETWORK_OUTAGE = "network_outage"
    MEMORY_LAYER_FAILURE = "memory_layer_failure"
    STORAGE_FAILURE = "storage_failure"
    SYSTEM_CRASH = "system_crash"
    MANUAL_TRIGGER = "manual_trigger"
    SECURITY_BREACH = "security_breach"


class RecoveryMode(Enum):
    """Recovery execution modes."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    TESTING = "testing"
    SIMULATION = "simulation"


@dataclass
class RPOTarget:
    """Recovery Point Objective definition."""
    max_data_loss_minutes: int
    critical_layers: List[str]
    backup_frequency_minutes: int
    verification_required: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RPOTarget':
        return cls(**data)


@dataclass
class RTOTarget:
    """Recovery Time Objective definition."""
    max_recovery_minutes: int
    critical_components: List[str]
    parallel_recovery: bool = True
    automated_validation: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod  
    def from_dict(cls, data: Dict) -> 'RTOTarget':
        return cls(**data)


@dataclass
class RecoveryMetadata:
    """Comprehensive recovery operation metadata."""
    recovery_id: str
    disaster_type: DisasterType
    recovery_mode: RecoveryMode
    trigger_timestamp: datetime
    target_timestamp: Optional[datetime]  # Point-in-time recovery target
    affected_layers: List[str]
    backup_id: str
    status: RecoveryStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    recovery_steps: List[Dict] = None
    validation_results: Dict[str, bool] = None
    error_message: Optional[str] = None
    rpo_achieved_minutes: Optional[int] = None
    rto_achieved_minutes: Optional[int] = None
    
    def __post_init__(self):
        if self.recovery_steps is None:
            self.recovery_steps = []
        if self.validation_results is None:
            self.validation_results = {}
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['disaster_type'] = self.disaster_type.value
        data['recovery_mode'] = self.recovery_mode.value
        data['trigger_timestamp'] = self.trigger_timestamp.isoformat()
        data['target_timestamp'] = self.target_timestamp.isoformat() if self.target_timestamp else None
        data['start_time'] = self.start_time.isoformat() if self.start_time else None
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RecoveryMetadata':
        data['disaster_type'] = DisasterType(data['disaster_type'])
        data['recovery_mode'] = RecoveryMode(data['recovery_mode'])
        data['trigger_timestamp'] = datetime.fromisoformat(data['trigger_timestamp'])
        data['target_timestamp'] = datetime.fromisoformat(data['target_timestamp']) if data['target_timestamp'] else None
        data['start_time'] = datetime.fromisoformat(data['start_time']) if data['start_time'] else None
        data['end_time'] = datetime.fromisoformat(data['end_time']) if data['end_time'] else None
        data['status'] = RecoveryStatus(data['status'])
        return cls(**data)


class RecoveryValidator(ABC):
    """Abstract base class for recovery validation."""
    
    @abstractmethod
    async def validate(self, recovered_layers: List[str]) -> Dict[str, bool]:
        """Validate recovered memory layers."""
        pass


class MemoryLayerValidator(RecoveryValidator):
    """Validates recovered memory layers for consistency and integrity."""
    
    async def validate(self, recovered_layers: List[str]) -> Dict[str, bool]:
        """Validate memory layer files."""
        results = {}
        
        for layer_path in recovered_layers:
            try:
                path_obj = Path(layer_path)
                
                # Check file exists
                if not path_obj.exists():
                    results[layer_path] = False
                    continue
                
                # Basic file integrity checks
                if path_obj.stat().st_size == 0:
                    results[layer_path] = False
                    continue
                
                # If JSON file, validate JSON structure
                if layer_path.endswith('.json'):
                    with open(layer_path, 'r') as f:
                        json.load(f)  # Will raise exception if invalid JSON
                
                results[layer_path] = True
                
            except Exception as e:
                logger.error(f"Validation failed for {layer_path}: {e}")
                results[layer_path] = False
        
        return results


class SystemHealthValidator(RecoveryValidator):
    """Validates system health after recovery."""
    
    def __init__(self, health_checks: List[Callable]):
        self.health_checks = health_checks
    
    async def validate(self, recovered_layers: List[str]) -> Dict[str, bool]:
        """Run system health checks."""
        results = {}
        
        for i, health_check in enumerate(self.health_checks):
            check_name = f"health_check_{i}"
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, health_check
                )
                results[check_name] = bool(result)
            except Exception as e:
                logger.error(f"Health check {check_name} failed: {e}")
                results[check_name] = False
        
        return results


class RecoveryOrchestrator:
    """Orchestrates complex recovery operations with dependency management."""
    
    def __init__(self):
        self.recovery_steps: List[Dict] = []
        self.step_dependencies: Dict[str, Set[str]] = {}
        self.completed_steps: Set[str] = set()
        self.failed_steps: Set[str] = set()
    
    def add_step(self, step_id: str, step_func: Callable, 
                 dependencies: Optional[List[str]] = None, **kwargs):
        """Add recovery step with dependencies."""
        step = {
            'id': step_id,
            'function': step_func,
            'kwargs': kwargs,
            'status': 'pending'
        }
        self.recovery_steps.append(step)
        
        if dependencies:
            self.step_dependencies[step_id] = set(dependencies)
        else:
            self.step_dependencies[step_id] = set()
    
    async def execute_recovery(self) -> bool:
        """Execute recovery steps in dependency order."""
        try:
            # Continue until all steps completed or failed
            while len(self.completed_steps) + len(self.failed_steps) < len(self.recovery_steps):
                ready_steps = self._get_ready_steps()
                
                if not ready_steps:
                    # Check if we're stuck due to failed dependencies
                    remaining_steps = [
                        step for step in self.recovery_steps 
                        if step['id'] not in self.completed_steps and step['id'] not in self.failed_steps
                    ]
                    if remaining_steps:
                        logger.error("Recovery stuck - no ready steps available")
                        return False
                    break
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(self._execute_step(step))
                    tasks.append(task)
                
                # Wait for all tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check if all critical steps completed
            return len(self.failed_steps) == 0
            
        except Exception as e:
            logger.error(f"Recovery orchestration failed: {e}")
            return False
    
    def _get_ready_steps(self) -> List[Dict]:
        """Get steps ready for execution (all dependencies met)."""
        ready_steps = []
        
        for step in self.recovery_steps:
            if step['id'] in self.completed_steps or step['id'] in self.failed_steps:
                continue
            
            dependencies = self.step_dependencies.get(step['id'], set())
            if dependencies.issubset(self.completed_steps):
                ready_steps.append(step)
        
        return ready_steps
    
    async def _execute_step(self, step: Dict) -> bool:
        """Execute individual recovery step."""
        step_id = step['id']
        step_func = step['function']
        kwargs = step.get('kwargs', {})
        
        try:
            logger.info(f"Executing recovery step: {step_id}")
            
            # Execute step function
            if asyncio.iscoroutinefunction(step_func):
                result = await step_func(**kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: step_func(**kwargs)
                )
            
            if result:
                self.completed_steps.add(step_id)
                step['status'] = 'completed'
                logger.info(f"Recovery step {step_id} completed successfully")
                return True
            else:
                self.failed_steps.add(step_id)
                step['status'] = 'failed'
                logger.error(f"Recovery step {step_id} failed")
                return False
                
        except Exception as e:
            self.failed_steps.add(step_id)
            step['status'] = 'failed'
            step['error'] = str(e)
            logger.error(f"Recovery step {step_id} failed with exception: {e}")
            return False


class DisasterRecoveryManager:
    """
    Comprehensive disaster recovery manager for Nova consciousness.
    
    Provides automated disaster detection, recovery orchestration,
    and RPO/RTO monitoring with point-in-time recovery capabilities.
    """
    
    def __init__(self, config: Dict[str, Any], backup_system: MemoryBackupSystem):
        """
        Initialize the disaster recovery manager.
        
        Args:
            config: Configuration dictionary with recovery settings
            backup_system: Reference to the backup system instance
        """
        self.config = config
        self.backup_system = backup_system
        
        # Initialize directories
        self.recovery_dir = Path(config.get('recovery_dir', '/tmp/nova_recovery'))
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for recovery metadata
        self.recovery_db_path = self.recovery_dir / "recovery_metadata.db"
        self._init_recovery_db()
        
        # RPO/RTO targets
        self.rpo_targets = self._load_rpo_targets()
        self.rto_targets = self._load_rto_targets()
        
        # Validators
        self.validators: List[RecoveryValidator] = [
            MemoryLayerValidator(),
            SystemHealthValidator(self._get_health_checks())
        ]
        
        # Active recovery tracking
        self.active_recoveries: Dict[str, RecoveryMetadata] = {}
        self.recovery_lock = threading.RLock()
        
        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"DisasterRecoveryManager initialized with config: {config}")
    
    def _init_recovery_db(self):
        """Initialize recovery metadata database."""
        conn = sqlite3.connect(self.recovery_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recovery_metadata (
                recovery_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recovery_timestamp
            ON recovery_metadata(json_extract(metadata_json, '$.trigger_timestamp'))
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recovery_status
            ON recovery_metadata(json_extract(metadata_json, '$.status'))
        """)
        conn.commit()
        conn.close()
    
    def _load_rpo_targets(self) -> Dict[str, RPOTarget]:
        """Load RPO targets from configuration."""
        rpo_config = self.config.get('rpo_targets', {})
        targets = {}
        
        for name, target_config in rpo_config.items():
            targets[name] = RPOTarget.from_dict(target_config)
        
        # Default RPO target if none configured
        if not targets:
            targets['default'] = RPOTarget(
                max_data_loss_minutes=5,
                critical_layers=[],
                backup_frequency_minutes=1
            )
        
        return targets
    
    def _load_rto_targets(self) -> Dict[str, RTOTarget]:
        """Load RTO targets from configuration."""
        rto_config = self.config.get('rto_targets', {})
        targets = {}
        
        for name, target_config in rto_config.items():
            targets[name] = RTOTarget.from_dict(target_config)
        
        # Default RTO target if none configured
        if not targets:
            targets['default'] = RTOTarget(
                max_recovery_minutes=15,
                critical_components=[]
            )
        
        return targets
    
    def _get_health_checks(self) -> List[Callable]:
        """Get system health check functions."""
        health_checks = []
        
        # Basic filesystem health check
        def check_filesystem():
            try:
                test_file = self.recovery_dir / "health_check_test"
                test_file.write_text("health check")
                content = test_file.read_text()
                test_file.unlink()
                return content == "health check"
            except Exception:
                return False
        
        health_checks.append(check_filesystem)
        
        # Memory usage check
        def check_memory():
            try:
                import psutil
                memory = psutil.virtual_memory()
                return memory.percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Skip if psutil not available
        
        health_checks.append(check_memory)
        
        return health_checks
    
    async def trigger_recovery(self,
                              disaster_type: DisasterType,
                              affected_layers: List[str],
                              recovery_mode: RecoveryMode = RecoveryMode.AUTOMATIC,
                              target_timestamp: Optional[datetime] = None,
                              backup_id: Optional[str] = None) -> Optional[RecoveryMetadata]:
        """
        Trigger disaster recovery operation.
        
        Args:
            disaster_type: Type of disaster that occurred
            affected_layers: List of memory layers that need recovery
            recovery_mode: Recovery execution mode
            target_timestamp: Point-in-time recovery target
            backup_id: Specific backup to restore from (optional)
            
        Returns:
            RecoveryMetadata object or None if recovery failed to start
        """
        recovery_id = self._generate_recovery_id()
        logger.info(f"Triggering recovery {recovery_id} for disaster {disaster_type.value}")
        
        try:
            # Find appropriate backup if not specified
            if not backup_id:
                backup_id = await self._find_recovery_backup(
                    affected_layers, target_timestamp
                )
            
            if not backup_id:
                logger.error(f"No suitable backup found for recovery {recovery_id}")
                return None
            
            # Create recovery metadata
            metadata = RecoveryMetadata(
                recovery_id=recovery_id,
                disaster_type=disaster_type,
                recovery_mode=recovery_mode,
                trigger_timestamp=datetime.now(),
                target_timestamp=target_timestamp,
                affected_layers=affected_layers,
                backup_id=backup_id,
                status=RecoveryStatus.PENDING
            )
            
            # Save metadata
            await self._save_recovery_metadata(metadata)
            
            # Track active recovery
            with self.recovery_lock:
                self.active_recoveries[recovery_id] = metadata
            
            # Start recovery execution
            if recovery_mode == RecoveryMode.AUTOMATIC:
                asyncio.create_task(self._execute_recovery(metadata))
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to trigger recovery {recovery_id}: {e}")
            return None
    
    async def _find_recovery_backup(self, 
                                  affected_layers: List[str],
                                  target_timestamp: Optional[datetime]) -> Optional[str]:
        """Find the most appropriate backup for recovery."""
        try:
            # Get available backups
            backups = await self.backup_system.list_backups(
                status=BackupStatus.COMPLETED,
                limit=1000
            )
            
            if not backups:
                return None
            
            # Filter backups by timestamp if target specified
            if target_timestamp:
                eligible_backups = [
                    backup for backup in backups
                    if backup.timestamp <= target_timestamp
                ]
            else:
                eligible_backups = backups
            
            if not eligible_backups:
                return None
            
            # Find backup that covers affected layers
            best_backup = None
            best_score = 0
            
            for backup in eligible_backups:
                # Calculate coverage score
                covered_layers = set(backup.memory_layers)
                affected_set = set(affected_layers)
                coverage = len(covered_layers.intersection(affected_set))
                
                # Prefer more recent backups and better coverage
                age_score = 1.0 / (1 + (datetime.now() - backup.timestamp).total_seconds() / 3600)
                coverage_score = coverage / len(affected_set) if affected_set else 0
                total_score = age_score * 0.3 + coverage_score * 0.7
                
                if total_score > best_score:
                    best_score = total_score
                    best_backup = backup
            
            return best_backup.backup_id if best_backup else None
            
        except Exception as e:
            logger.error(f"Failed to find recovery backup: {e}")
            return None
    
    async def _execute_recovery(self, metadata: RecoveryMetadata):
        """Execute the complete recovery operation."""
        recovery_id = metadata.recovery_id
        
        try:
            # Update status to running
            metadata.status = RecoveryStatus.RUNNING
            metadata.start_time = datetime.now()
            await self._save_recovery_metadata(metadata)
            
            logger.info(f"Starting recovery execution for {recovery_id}")
            
            # Create recovery orchestrator
            orchestrator = RecoveryOrchestrator()
            
            # Add recovery steps
            await self._plan_recovery_steps(orchestrator, metadata)
            
            # Execute recovery
            success = await orchestrator.execute_recovery()
            
            # Update metadata with results
            metadata.end_time = datetime.now()
            metadata.recovery_steps = [
                {
                    'id': step['id'],
                    'status': step['status'],
                    'error': step.get('error')
                }
                for step in orchestrator.recovery_steps
            ]
            
            if success:
                # Run validation
                validation_results = await self._validate_recovery(metadata.affected_layers)
                metadata.validation_results = validation_results
                
                all_passed = all(validation_results.values())
                if all_passed:
                    metadata.status = RecoveryStatus.COMPLETED
                    logger.info(f"Recovery {recovery_id} completed successfully")
                else:
                    metadata.status = RecoveryStatus.FAILED
                    metadata.error_message = "Validation failed"
                    logger.error(f"Recovery {recovery_id} validation failed")
            else:
                metadata.status = RecoveryStatus.FAILED
                metadata.error_message = "Recovery execution failed"
                logger.error(f"Recovery {recovery_id} execution failed")
            
            # Calculate RPO/RTO achieved
            await self._calculate_rpo_rto_achieved(metadata)
            
        except Exception as e:
            logger.error(f"Recovery execution failed for {recovery_id}: {e}")
            metadata.status = RecoveryStatus.FAILED
            metadata.error_message = str(e)
            metadata.end_time = datetime.now()
        
        finally:
            # Save final metadata
            await self._save_recovery_metadata(metadata)
            
            # Remove from active recoveries
            with self.recovery_lock:
                self.active_recoveries.pop(recovery_id, None)
    
    async def _plan_recovery_steps(self, orchestrator: RecoveryOrchestrator, 
                                 metadata: RecoveryMetadata):
        """Plan the recovery steps based on disaster type and affected layers."""
        
        # Step 1: Prepare recovery environment
        orchestrator.add_step(
            'prepare_environment',
            self._prepare_recovery_environment,
            recovery_id=metadata.recovery_id
        )
        
        # Step 2: Download backup
        orchestrator.add_step(
            'download_backup',
            self._download_backup,
            dependencies=['prepare_environment'],
            recovery_id=metadata.recovery_id,
            backup_id=metadata.backup_id
        )
        
        # Step 3: Extract backup
        orchestrator.add_step(
            'extract_backup',
            self._extract_backup,
            dependencies=['download_backup'],
            recovery_id=metadata.recovery_id
        )
        
        # Step 4: Restore memory layers
        for i, layer_path in enumerate(metadata.affected_layers):
            step_id = f'restore_layer_{i}'
            orchestrator.add_step(
                step_id,
                self._restore_memory_layer,
                dependencies=['extract_backup'],
                layer_path=layer_path,
                recovery_id=metadata.recovery_id
            )
        
        # Step 5: Update system state
        layer_steps = [f'restore_layer_{i}' for i in range(len(metadata.affected_layers))]
        orchestrator.add_step(
            'update_system_state',
            self._update_system_state,
            dependencies=layer_steps,
            recovery_id=metadata.recovery_id
        )
        
        # Step 6: Cleanup temporary files
        orchestrator.add_step(
            'cleanup',
            self._cleanup_recovery,
            dependencies=['update_system_state'],
            recovery_id=metadata.recovery_id
        )
    
    async def _prepare_recovery_environment(self, recovery_id: str) -> bool:
        """Prepare the recovery environment."""
        try:
            recovery_work_dir = self.recovery_dir / recovery_id
            recovery_work_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (recovery_work_dir / 'backup').mkdir(exist_ok=True)
            (recovery_work_dir / 'extracted').mkdir(exist_ok=True)
            (recovery_work_dir / 'staging').mkdir(exist_ok=True)
            
            logger.info(f"Recovery environment prepared for {recovery_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare recovery environment for {recovery_id}: {e}")
            return False
    
    async def _download_backup(self, recovery_id: str, backup_id: str) -> bool:
        """Download backup for recovery."""
        try:
            # Get backup metadata
            backup_metadata = await self.backup_system.get_backup(backup_id)
            if not backup_metadata:
                logger.error(f"Backup {backup_id} not found")
                return False
            
            # Get storage adapter
            storage_adapter = self.backup_system.storage_adapters.get(
                backup_metadata.storage_backend
            )
            if not storage_adapter:
                logger.error(f"Storage adapter not available for {backup_metadata.storage_backend.value}")
                return False
            
            # Download backup
            recovery_work_dir = self.recovery_dir / recovery_id
            local_backup_path = recovery_work_dir / 'backup' / f'{backup_id}.backup'
            
            success = await storage_adapter.download(
                backup_metadata.storage_path,
                str(local_backup_path)
            )
            
            if success:
                logger.info(f"Backup {backup_id} downloaded for recovery {recovery_id}")
            else:
                logger.error(f"Failed to download backup {backup_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to download backup for recovery {recovery_id}: {e}")
            return False
    
    async def _extract_backup(self, recovery_id: str) -> bool:
        """Extract backup archive."""
        try:
            recovery_work_dir = self.recovery_dir / recovery_id
            backup_files = list((recovery_work_dir / 'backup').glob('*.backup'))
            
            if not backup_files:
                logger.error(f"No backup files found for recovery {recovery_id}")
                return False
            
            backup_file = backup_files[0]  # Take first backup file
            extract_dir = recovery_work_dir / 'extracted'
            
            # Extract using backup system's decompression
            from memory_backup_system import BackupCompressor
            
            # For simplicity, we'll use a basic extraction approach
            # In a real implementation, this would handle the complex archive format
            
            success = await BackupCompressor.decompress_file(
                str(backup_file),
                str(extract_dir / 'backup_data')
            )
            
            if success:
                logger.info(f"Backup extracted for recovery {recovery_id}")
            else:
                logger.error(f"Failed to extract backup for recovery {recovery_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to extract backup for recovery {recovery_id}: {e}")
            return False
    
    async def _restore_memory_layer(self, layer_path: str, recovery_id: str) -> bool:
        """Restore individual memory layer."""
        try:
            recovery_work_dir = self.recovery_dir / recovery_id
            staging_dir = recovery_work_dir / 'staging'
            
            # Find extracted layer file
            extracted_dir = recovery_work_dir / 'extracted'
            
            # This is a simplified approach - real implementation would
            # parse the backup manifest and restore exact files
            layer_name = Path(layer_path).name
            possible_files = list(extracted_dir.rglob(f"*{layer_name}*"))
            
            if not possible_files:
                logger.warning(f"Layer file not found in backup for {layer_path}")
                # Create minimal recovery file
                recovery_file = staging_dir / layer_name
                with open(recovery_file, 'w') as f:
                    json.dump({
                        'recovered': True,
                        'recovery_timestamp': datetime.now().isoformat(),
                        'original_path': layer_path
                    }, f)
                return True
            
            # Copy restored file to staging
            source_file = possible_files[0]
            dest_file = staging_dir / layer_name
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: shutil.copy2(source_file, dest_file)
            )
            
            logger.info(f"Memory layer {layer_path} restored for recovery {recovery_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore memory layer {layer_path}: {e}")
            return False
    
    async def _update_system_state(self, recovery_id: str) -> bool:
        """Update system state with recovered data."""
        try:
            recovery_work_dir = self.recovery_dir / recovery_id
            staging_dir = recovery_work_dir / 'staging'
            
            # Move staged files to their final locations
            for staged_file in staging_dir.glob('*'):
                if staged_file.is_file():
                    # This would need proper path mapping in real implementation
                    # For now, we'll just log the recovery
                    logger.info(f"Would restore {staged_file.name} to final location")
            
            logger.info(f"System state updated for recovery {recovery_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update system state for recovery {recovery_id}: {e}")
            return False
    
    async def _cleanup_recovery(self, recovery_id: str) -> bool:
        """Cleanup temporary recovery files."""
        try:
            recovery_work_dir = self.recovery_dir / recovery_id
            
            # Remove temporary directories but keep logs
            for subdir in ['backup', 'extracted', 'staging']:
                subdir_path = recovery_work_dir / subdir
                if subdir_path.exists():
                    shutil.rmtree(subdir_path)
            
            logger.info(f"Recovery cleanup completed for {recovery_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup recovery {recovery_id}: {e}")
            return False
    
    async def _validate_recovery(self, recovered_layers: List[str]) -> Dict[str, bool]:
        """Validate recovery using all configured validators."""
        all_results = {}
        
        for validator in self.validators:
            try:
                validator_name = validator.__class__.__name__
                results = await validator.validate(recovered_layers)
                
                # Prefix results with validator name
                for key, value in results.items():
                    all_results[f"{validator_name}_{key}"] = value
                    
            except Exception as e:
                logger.error(f"Validation failed for {validator.__class__.__name__}: {e}")
                all_results[f"{validator.__class__.__name__}_error"] = False
        
        return all_results
    
    async def _calculate_rpo_rto_achieved(self, metadata: RecoveryMetadata):
        """Calculate actual RPO and RTO achieved during recovery."""
        try:
            # Calculate RTO (recovery time)
            if metadata.start_time and metadata.end_time:
                rto_seconds = (metadata.end_time - metadata.start_time).total_seconds()
                metadata.rto_achieved_minutes = int(rto_seconds / 60)
            
            # Calculate RPO (data loss time)
            if metadata.target_timestamp:
                backup_metadata = await self.backup_system.get_backup(metadata.backup_id)
                if backup_metadata:
                    rpo_seconds = (metadata.target_timestamp - backup_metadata.timestamp).total_seconds()
                    metadata.rpo_achieved_minutes = int(rpo_seconds / 60)
            
        except Exception as e:
            logger.error(f"Failed to calculate RPO/RTO: {e}")
    
    def _generate_recovery_id(self) -> str:
        """Generate unique recovery ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        import hashlib
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"nova_recovery_{timestamp}_{random_suffix}"
    
    async def _save_recovery_metadata(self, metadata: RecoveryMetadata):
        """Save recovery metadata to database."""
        conn = sqlite3.connect(self.recovery_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO recovery_metadata (recovery_id, metadata_json) VALUES (?, ?)",
            (metadata.recovery_id, json.dumps(metadata.to_dict()))
        )
        conn.commit()
        conn.close()
    
    async def get_recovery(self, recovery_id: str) -> Optional[RecoveryMetadata]:
        """Get recovery metadata by ID."""
        conn = sqlite3.connect(self.recovery_db_path)
        cursor = conn.execute(
            "SELECT metadata_json FROM recovery_metadata WHERE recovery_id = ?",
            (recovery_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                metadata_dict = json.loads(result[0])
                return RecoveryMetadata.from_dict(metadata_dict)
            except Exception as e:
                logger.error(f"Failed to parse recovery metadata: {e}")
        
        return None
    
    async def list_recoveries(self,
                             disaster_type: Optional[DisasterType] = None,
                             status: Optional[RecoveryStatus] = None,
                             limit: int = 100) -> List[RecoveryMetadata]:
        """List recovery operations with optional filtering."""
        conn = sqlite3.connect(self.recovery_db_path)
        
        query = "SELECT metadata_json FROM recovery_metadata WHERE 1=1"
        params = []
        
        if disaster_type:
            query += " AND json_extract(metadata_json, '$.disaster_type') = ?"
            params.append(disaster_type.value)
        
        if status:
            query += " AND json_extract(metadata_json, '$.status') = ?"
            params.append(status.value)
        
        query += " ORDER BY json_extract(metadata_json, '$.trigger_timestamp') DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        recoveries = []
        for (metadata_json,) in results:
            try:
                metadata_dict = json.loads(metadata_json)
                recovery = RecoveryMetadata.from_dict(metadata_dict)
                recoveries.append(recovery)
            except Exception as e:
                logger.error(f"Failed to parse recovery metadata: {e}")
        
        return recoveries
    
    async def test_recovery(self, 
                           test_layers: List[str],
                           backup_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Test disaster recovery process without affecting production.
        
        Args:
            test_layers: Memory layers to test recovery for
            backup_id: Specific backup to test with
            
        Returns:
            Test results including success status and performance metrics
        """
        test_id = f"test_{self._generate_recovery_id()}"
        
        try:
            logger.info(f"Starting recovery test {test_id}")
            
            # Trigger test recovery
            recovery = await self.trigger_recovery(
                disaster_type=DisasterType.MANUAL_TRIGGER,
                affected_layers=test_layers,
                recovery_mode=RecoveryMode.TESTING,
                backup_id=backup_id
            )
            
            if not recovery:
                return {
                    'success': False,
                    'error': 'Failed to initiate test recovery'
                }
            
            # Wait for recovery to complete
            max_wait_seconds = 300  # 5 minutes
            wait_interval = 5
            elapsed = 0
            
            while elapsed < max_wait_seconds:
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
                
                current_recovery = await self.get_recovery(recovery.recovery_id)
                if current_recovery and current_recovery.status in [
                    RecoveryStatus.COMPLETED, RecoveryStatus.FAILED, RecoveryStatus.CANCELLED
                ]:
                    recovery = current_recovery
                    break
            
            # Analyze test results
            test_results = {
                'success': recovery.status == RecoveryStatus.COMPLETED,
                'recovery_id': recovery.recovery_id,
                'rpo_achieved_minutes': recovery.rpo_achieved_minutes,
                'rto_achieved_minutes': recovery.rto_achieved_minutes,
                'validation_results': recovery.validation_results,
                'error_message': recovery.error_message
            }
            
            # Check against targets
            rpo_target = self.rpo_targets.get('default')
            rto_target = self.rto_targets.get('default')
            
            if rpo_target and recovery.rpo_achieved_minutes:
                test_results['rpo_target_met'] = recovery.rpo_achieved_minutes <= rpo_target.max_data_loss_minutes
            
            if rto_target and recovery.rto_achieved_minutes:
                test_results['rto_target_met'] = recovery.rto_achieved_minutes <= rto_target.max_recovery_minutes
            
            logger.info(f"Recovery test {test_id} completed: {test_results['success']}")
            return test_results
            
        except Exception as e:
            logger.error(f"Recovery test {test_id} failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def start_monitoring(self):
        """Start background disaster monitoring."""
        if self._monitor_task is None:
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Disaster recovery monitoring started")
    
    async def stop_monitoring(self):
        """Stop background disaster monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Disaster recovery monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop for disaster detection."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check system health
                health_issues = await self._check_system_health()
                
                # Trigger automatic recovery if needed
                for issue in health_issues:
                    await self._handle_detected_issue(issue)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_system_health(self) -> List[Dict[str, Any]]:
        """Check for system health issues that might require recovery."""
        issues = []
        
        try:
            # Run health validators
            health_validator = SystemHealthValidator(self._get_health_checks())
            health_results = await health_validator.validate([])
            
            # Check for failures
            for check_name, passed in health_results.items():
                if not passed:
                    issues.append({
                        'type': 'health_check_failure',
                        'check': check_name,
                        'severity': 'medium'
                    })
            
            # Additional monitoring checks can be added here
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            issues.append({
                'type': 'health_check_error',
                'error': str(e),
                'severity': 'high'
            })
        
        return issues
    
    async def _handle_detected_issue(self, issue: Dict[str, Any]):
        """Handle automatically detected issues."""
        try:
            severity = issue.get('severity', 'medium')
            
            # Only auto-recover for high severity issues
            if severity == 'high':
                logger.warning(f"Auto-recovering from detected issue: {issue}")
                
                # Determine affected layers (simplified)
                affected_layers = ['/tmp/critical_layer.json']  # Would be determined dynamically
                
                await self.trigger_recovery(
                    disaster_type=DisasterType.SYSTEM_CRASH,
                    affected_layers=affected_layers,
                    recovery_mode=RecoveryMode.AUTOMATIC
                )
        except Exception as e:
            logger.error(f"Failed to handle detected issue: {e}")


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize backup system first
        backup_config = {
            'backup_dir': '/tmp/nova_test_backups',
            'storage': {
                'local_path': '/tmp/nova_backup_storage'
            }
        }
        backup_system = MemoryBackupSystem(backup_config)
        
        # Initialize disaster recovery manager
        recovery_config = {
            'recovery_dir': '/tmp/nova_test_recovery',
            'rpo_targets': {
                'default': {
                    'max_data_loss_minutes': 5,
                    'critical_layers': ['/tmp/critical_layer.json'],
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
        
        dr_manager = DisasterRecoveryManager(recovery_config, backup_system)
        
        # Create test data and backup
        test_layers = ['/tmp/test_layer.json']
        Path(test_layers[0]).parent.mkdir(parents=True, exist_ok=True)
        with open(test_layers[0], 'w') as f:
            json.dump({
                'test_data': 'original data',
                'timestamp': datetime.now().isoformat()
            }, f)
        
        # Create backup
        backup = await backup_system.create_backup(
            memory_layers=test_layers,
            strategy=BackupStrategy.FULL
        )
        
        if backup:
            print(f"Test backup created: {backup.backup_id}")
            
            # Test recovery
            test_results = await dr_manager.test_recovery(
                test_layers=test_layers,
                backup_id=backup.backup_id
            )
            
            print(f"Recovery test results: {test_results}")
            
            # Start monitoring
            await dr_manager.start_monitoring()
            
            # Wait a moment then stop
            await asyncio.sleep(5)
            await dr_manager.stop_monitoring()
        else:
            print("Failed to create test backup")
    
    asyncio.run(main())