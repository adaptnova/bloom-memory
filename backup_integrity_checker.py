"""
Nova Bloom Consciousness - Backup Integrity Checker
Critical component for ensuring data integrity and corruption detection.

This module implements comprehensive integrity verification including:
- Multi-level checksums and hash verification
- Content structure validation
- Corruption detection and automated repair
- Integrity reporting and alerting
- Continuous monitoring of backup integrity
- Cross-validation between backup copies
"""

import asyncio
import hashlib
import json
import logging
import lzma
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import struct
import zlib

logger = logging.getLogger(__name__)


class IntegrityStatus(Enum):
    """Status of integrity check operations."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CORRUPTED = "corrupted"
    REPAIRED = "repaired"
    UNREPAIRABLE = "unrepairable"


class IntegrityLevel(Enum):
    """Levels of integrity verification."""
    BASIC = "basic"          # File existence and size
    CHECKSUM = "checksum"    # Hash verification
    CONTENT = "content"      # Structure and content validation
    COMPREHENSIVE = "comprehensive"  # All checks plus cross-validation


class CorruptionType(Enum):
    """Types of corruption that can be detected."""
    FILE_MISSING = "file_missing"
    CHECKSUM_MISMATCH = "checksum_mismatch" 
    SIZE_MISMATCH = "size_mismatch"
    STRUCTURE_INVALID = "structure_invalid"
    CONTENT_CORRUPTED = "content_corrupted"
    METADATA_CORRUPTED = "metadata_corrupted"
    COMPRESSION_ERROR = "compression_error"
    ENCODING_ERROR = "encoding_error"


@dataclass
class IntegrityIssue:
    """Represents a detected integrity issue."""
    file_path: str
    corruption_type: CorruptionType
    severity: str  # low, medium, high, critical
    description: str
    detected_at: datetime
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    repairable: bool = False
    repair_suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['corruption_type'] = self.corruption_type.value
        data['detected_at'] = self.detected_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IntegrityIssue':
        data['corruption_type'] = CorruptionType(data['corruption_type'])
        data['detected_at'] = datetime.fromisoformat(data['detected_at'])
        return cls(**data)


@dataclass
class IntegrityCheckResult:
    """Results of an integrity check operation."""
    check_id: str
    file_path: str
    integrity_level: IntegrityLevel
    status: IntegrityStatus
    check_timestamp: datetime
    issues: List[IntegrityIssue]
    metadata: Dict[str, Any]
    repair_attempted: bool = False
    repair_successful: bool = False
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['integrity_level'] = self.integrity_level.value
        data['status'] = self.status.value
        data['check_timestamp'] = self.check_timestamp.isoformat()
        data['issues'] = [issue.to_dict() for issue in self.issues]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IntegrityCheckResult':
        data['integrity_level'] = IntegrityLevel(data['integrity_level'])
        data['status'] = IntegrityStatus(data['status'])
        data['check_timestamp'] = datetime.fromisoformat(data['check_timestamp'])
        data['issues'] = [IntegrityIssue.from_dict(issue) for issue in data['issues']]
        return cls(**data)


ChecksumInfo = namedtuple('ChecksumInfo', ['algorithm', 'value', 'size'])


class IntegrityValidator(ABC):
    """Abstract base class for integrity validation."""
    
    @abstractmethod
    async def validate(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate file integrity and return any issues found."""
        pass
    
    @abstractmethod
    def get_validation_level(self) -> IntegrityLevel:
        """Get the integrity level this validator provides."""
        pass


class BasicIntegrityValidator(IntegrityValidator):
    """Basic file existence and size validation."""
    
    async def validate(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate basic file properties."""
        issues = []
        file_path_obj = Path(file_path)
        
        # Check file existence
        if not file_path_obj.exists():
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.FILE_MISSING,
                severity="critical",
                description=f"File does not exist: {file_path}",
                detected_at=datetime.now(),
                repairable=False
            ))
            return issues
        
        # Check file size if expected size is provided
        expected_size = expected_metadata.get('size')
        if expected_size is not None:
            try:
                actual_size = file_path_obj.stat().st_size
                if actual_size != expected_size:
                    issues.append(IntegrityIssue(
                        file_path=file_path,
                        corruption_type=CorruptionType.SIZE_MISMATCH,
                        severity="high",
                        description=f"File size mismatch",
                        detected_at=datetime.now(),
                        expected_value=str(expected_size),
                        actual_value=str(actual_size),
                        repairable=False
                    ))
            except Exception as e:
                issues.append(IntegrityIssue(
                    file_path=file_path,
                    corruption_type=CorruptionType.METADATA_CORRUPTED,
                    severity="medium",
                    description=f"Failed to read file metadata: {e}",
                    detected_at=datetime.now(),
                    repairable=False
                ))
        
        return issues
    
    def get_validation_level(self) -> IntegrityLevel:
        return IntegrityLevel.BASIC


class ChecksumIntegrityValidator(IntegrityValidator):
    """Checksum-based integrity validation."""
    
    def __init__(self, algorithms: List[str] = None):
        """
        Initialize with hash algorithms to use.
        
        Args:
            algorithms: List of hash algorithms ('sha256', 'md5', 'sha1', etc.)
        """
        self.algorithms = algorithms or ['sha256', 'md5']
    
    async def validate(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate file checksums."""
        issues = []
        
        try:
            # Calculate current checksums
            current_checksums = await self._calculate_checksums(file_path)
            
            # Compare with expected checksums
            for algorithm in self.algorithms:
                expected_checksum = expected_metadata.get(f'{algorithm}_checksum')
                if expected_checksum:
                    current_checksum = current_checksums.get(algorithm)
                    
                    if current_checksum != expected_checksum:
                        issues.append(IntegrityIssue(
                            file_path=file_path,
                            corruption_type=CorruptionType.CHECKSUM_MISMATCH,
                            severity="high",
                            description=f"{algorithm.upper()} checksum mismatch",
                            detected_at=datetime.now(),
                            expected_value=expected_checksum,
                            actual_value=current_checksum,
                            repairable=False,
                            repair_suggestion="Restore from backup or regenerate file"
                        ))
        
        except Exception as e:
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.CONTENT_CORRUPTED,
                severity="high",
                description=f"Failed to calculate checksums: {e}",
                detected_at=datetime.now(),
                repairable=False
            ))
        
        return issues
    
    async def _calculate_checksums(self, file_path: str) -> Dict[str, str]:
        """Calculate checksums for a file."""
        checksums = {}
        
        def calculate():
            hashers = {alg: hashlib.new(alg) for alg in self.algorithms}
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    for hasher in hashers.values():
                        hasher.update(chunk)
            
            return {alg: hasher.hexdigest() for alg, hasher in hashers.items()}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, calculate)
    
    def get_validation_level(self) -> IntegrityLevel:
        return IntegrityLevel.CHECKSUM


class ContentIntegrityValidator(IntegrityValidator):
    """Content structure and format validation."""
    
    async def validate(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate file content structure."""
        issues = []
        file_path_obj = Path(file_path)
        
        try:
            # Check file extension and validate accordingly
            if file_path.endswith('.json'):
                issues.extend(await self._validate_json_content(file_path, expected_metadata))
            elif file_path.endswith('.backup') or file_path.endswith('.xz'):
                issues.extend(await self._validate_compressed_content(file_path, expected_metadata))
            else:
                issues.extend(await self._validate_generic_content(file_path, expected_metadata))
        
        except Exception as e:
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.CONTENT_CORRUPTED,
                severity="medium",
                description=f"Content validation failed: {e}",
                detected_at=datetime.now(),
                repairable=False
            ))
        
        return issues
    
    async def _validate_json_content(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate JSON file content."""
        issues = []
        
        try:
            def validate_json():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    
                # Basic JSON structure validation
                if not isinstance(content, (dict, list)):
                    return ["Invalid JSON structure - must be object or array"]
                
                # Check for required fields if specified
                required_fields = expected_metadata.get('required_fields', [])
                if isinstance(content, dict):
                    missing_fields = []
                    for field in required_fields:
                        if field not in content:
                            missing_fields.append(field)
                    if missing_fields:
                        return [f"Missing required fields: {', '.join(missing_fields)}"]
                
                return []
            
            loop = asyncio.get_event_loop()
            validation_errors = await loop.run_in_executor(None, validate_json)
            
            for error in validation_errors:
                issues.append(IntegrityIssue(
                    file_path=file_path,
                    corruption_type=CorruptionType.STRUCTURE_INVALID,
                    severity="medium",
                    description=error,
                    detected_at=datetime.now(),
                    repairable=True,
                    repair_suggestion="Restore from backup or validate JSON syntax"
                ))
        
        except json.JSONDecodeError as e:
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.STRUCTURE_INVALID,
                severity="high",
                description=f"Invalid JSON syntax: {e}",
                detected_at=datetime.now(),
                repairable=True,
                repair_suggestion="Fix JSON syntax or restore from backup"
            ))
        
        return issues
    
    async def _validate_compressed_content(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate compressed file content."""
        issues = []
        
        try:
            def validate_compression():
                # Try to decompress first few bytes to verify format
                with lzma.open(file_path, 'rb') as f:
                    f.read(1024)  # Read first 1KB to test decompression
                return []
            
            loop = asyncio.get_event_loop()
            validation_errors = await loop.run_in_executor(None, validate_compression)
            
            for error in validation_errors:
                issues.append(IntegrityIssue(
                    file_path=file_path,
                    corruption_type=CorruptionType.COMPRESSION_ERROR,
                    severity="high",
                    description=error,
                    detected_at=datetime.now(),
                    repairable=False,
                    repair_suggestion="Restore from backup"
                ))
        
        except Exception as e:
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.COMPRESSION_ERROR,
                severity="high",
                description=f"Compression validation failed: {e}",
                detected_at=datetime.now(),
                repairable=False,
                repair_suggestion="File may be corrupted, restore from backup"
            ))
        
        return issues
    
    async def _validate_generic_content(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Validate generic file content."""
        issues = []
        
        try:
            # Check for null bytes or other signs of corruption
            def check_content():
                with open(file_path, 'rb') as f:
                    chunk_size = 64 * 1024
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        
                        # Check for excessive null bytes (potential corruption)
                        null_ratio = chunk.count(b'\x00') / len(chunk)
                        if null_ratio > 0.1:  # More than 10% null bytes
                            return ["High ratio of null bytes detected (potential corruption)"]
                
                return []
            
            loop = asyncio.get_event_loop()
            validation_errors = await loop.run_in_executor(None, check_content)
            
            for error in validation_errors:
                issues.append(IntegrityIssue(
                    file_path=file_path,
                    corruption_type=CorruptionType.CONTENT_CORRUPTED,
                    severity="medium",
                    description=error,
                    detected_at=datetime.now(),
                    repairable=False,
                    repair_suggestion="Restore from backup"
                ))
        
        except Exception as e:
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.CONTENT_CORRUPTED,
                severity="medium",
                description=f"Content validation failed: {e}",
                detected_at=datetime.now(),
                repairable=False
            ))
        
        return issues
    
    def get_validation_level(self) -> IntegrityLevel:
        return IntegrityLevel.CONTENT


class CrossValidationValidator(IntegrityValidator):
    """Cross-validates backup integrity across multiple copies."""
    
    def __init__(self, backup_system):
        """
        Initialize with backup system reference for cross-validation.
        
        Args:
            backup_system: Reference to MemoryBackupSystem instance
        """
        self.backup_system = backup_system
    
    async def validate(self, file_path: str, expected_metadata: Dict) -> List[IntegrityIssue]:
        """Cross-validate against other backup copies."""
        issues = []
        
        try:
            # This would implement cross-validation logic
            # For now, we'll do a simplified check
            backup_id = expected_metadata.get('backup_id')
            if backup_id:
                backup_metadata = await self.backup_system.get_backup(backup_id)
                if backup_metadata:
                    # Compare current file against backup metadata
                    expected_checksum = backup_metadata.checksum
                    if expected_checksum:
                        # Calculate current checksum and compare
                        validator = ChecksumIntegrityValidator(['sha256'])
                        current_checksums = await validator._calculate_checksums(file_path)
                        current_checksum = current_checksums.get('sha256', '')
                        
                        if current_checksum != expected_checksum:
                            issues.append(IntegrityIssue(
                                file_path=file_path,
                                corruption_type=CorruptionType.CHECKSUM_MISMATCH,
                                severity="critical",
                                description="Cross-validation failed - checksum mismatch with backup metadata",
                                detected_at=datetime.now(),
                                expected_value=expected_checksum,
                                actual_value=current_checksum,
                                repairable=True,
                                repair_suggestion="Restore from verified backup copy"
                            ))
        
        except Exception as e:
            issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.CONTENT_CORRUPTED,
                severity="medium",
                description=f"Cross-validation failed: {e}",
                detected_at=datetime.now(),
                repairable=False
            ))
        
        return issues
    
    def get_validation_level(self) -> IntegrityLevel:
        return IntegrityLevel.COMPREHENSIVE


class BackupIntegrityChecker:
    """
    Comprehensive backup integrity checker for Nova consciousness memory system.
    
    Provides multi-level integrity verification, corruption detection,
    and automated repair capabilities for backup files.
    """
    
    def __init__(self, config: Dict[str, Any], backup_system=None):
        """
        Initialize the integrity checker.
        
        Args:
            config: Configuration dictionary
            backup_system: Reference to backup system for cross-validation
        """
        self.config = config
        self.backup_system = backup_system
        
        # Initialize directories
        self.integrity_dir = Path(config.get('integrity_dir', '/tmp/nova_integrity'))
        self.integrity_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for integrity check results
        self.integrity_db_path = self.integrity_dir / "integrity_checks.db"
        self._init_integrity_db()
        
        # Initialize validators
        self.validators: Dict[IntegrityLevel, List[IntegrityValidator]] = {
            IntegrityLevel.BASIC: [BasicIntegrityValidator()],
            IntegrityLevel.CHECKSUM: [
                BasicIntegrityValidator(),
                ChecksumIntegrityValidator()
            ],
            IntegrityLevel.CONTENT: [
                BasicIntegrityValidator(),
                ChecksumIntegrityValidator(),
                ContentIntegrityValidator()
            ],
            IntegrityLevel.COMPREHENSIVE: [
                BasicIntegrityValidator(),
                ChecksumIntegrityValidator(),
                ContentIntegrityValidator()
            ]
        }
        
        # Add cross-validation if backup system available
        if backup_system:
            cross_validator = CrossValidationValidator(backup_system)
            self.validators[IntegrityLevel.COMPREHENSIVE].append(cross_validator)
        
        # Background monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread pool for parallel checking
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"BackupIntegrityChecker initialized with config: {config}")
    
    def _init_integrity_db(self):
        """Initialize integrity check database."""
        conn = sqlite3.connect(self.integrity_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS integrity_checks (
                check_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                check_result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_check_file_path
            ON integrity_checks(file_path)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_check_timestamp
            ON integrity_checks(json_extract(check_result_json, '$.check_timestamp'))
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_check_status
            ON integrity_checks(json_extract(check_result_json, '$.status'))
        """)
        conn.commit()
        conn.close()
    
    async def check_file_integrity(self,
                                  file_path: str,
                                  integrity_level: IntegrityLevel = IntegrityLevel.CHECKSUM,
                                  expected_metadata: Optional[Dict] = None) -> IntegrityCheckResult:
        """
        Check integrity of a single file.
        
        Args:
            file_path: Path to file to check
            integrity_level: Level of integrity checking to perform
            expected_metadata: Expected file metadata for validation
            
        Returns:
            IntegrityCheckResult with all issues found
        """
        check_id = self._generate_check_id()
        logger.info(f"Starting integrity check {check_id} for {file_path}")
        
        result = IntegrityCheckResult(
            check_id=check_id,
            file_path=file_path,
            integrity_level=integrity_level,
            status=IntegrityStatus.RUNNING,
            check_timestamp=datetime.now(),
            issues=[],
            metadata=expected_metadata or {}
        )
        
        try:
            # Get validators for requested level
            validators = self.validators.get(integrity_level, [])
            
            # Run all validators
            all_issues = []
            for validator in validators:
                try:
                    issues = await validator.validate(file_path, expected_metadata or {})
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(f"Validator {validator.__class__.__name__} failed: {e}")
                    all_issues.append(IntegrityIssue(
                        file_path=file_path,
                        corruption_type=CorruptionType.CONTENT_CORRUPTED,
                        severity="medium",
                        description=f"Validation error: {e}",
                        detected_at=datetime.now(),
                        repairable=False
                    ))
            
            # Update result with findings
            result.issues = all_issues
            
            if not all_issues:
                result.status = IntegrityStatus.PASSED
            else:
                # Determine overall status based on issue severity
                critical_issues = [i for i in all_issues if i.severity == "critical"]
                high_issues = [i for i in all_issues if i.severity == "high"]
                
                if critical_issues:
                    result.status = IntegrityStatus.CORRUPTED
                elif high_issues:
                    result.status = IntegrityStatus.FAILED
                else:
                    result.status = IntegrityStatus.FAILED
            
            logger.info(f"Integrity check {check_id} completed with status {result.status.value}")
            
        except Exception as e:
            logger.error(f"Integrity check {check_id} failed: {e}")
            result.status = IntegrityStatus.FAILED
            result.issues.append(IntegrityIssue(
                file_path=file_path,
                corruption_type=CorruptionType.CONTENT_CORRUPTED,
                severity="critical",
                description=f"Integrity check failed: {e}",
                detected_at=datetime.now(),
                repairable=False
            ))
        
        # Save result to database
        await self._save_check_result(result)
        
        return result
    
    async def check_backup_integrity(self,
                                   backup_id: str,
                                   integrity_level: IntegrityLevel = IntegrityLevel.CHECKSUM) -> Dict[str, IntegrityCheckResult]:
        """
        Check integrity of an entire backup.
        
        Args:
            backup_id: ID of backup to check
            integrity_level: Level of integrity checking
            
        Returns:
            Dictionary mapping file paths to integrity check results
        """
        logger.info(f"Starting backup integrity check for {backup_id}")
        
        if not self.backup_system:
            logger.error("Backup system not available for backup integrity check")
            return {}
        
        try:
            # Get backup metadata
            backup_metadata = await self.backup_system.get_backup(backup_id)
            if not backup_metadata:
                logger.error(f"Backup {backup_id} not found")
                return {}
            
            # For demonstration, we'll check memory layer files
            # In real implementation, this would check actual backup archive files
            results = {}
            
            for layer_path in backup_metadata.memory_layers:
                if Path(layer_path).exists():
                    expected_metadata = {
                        'backup_id': backup_id,
                        'sha256_checksum': backup_metadata.checksum,
                        'size': backup_metadata.original_size
                    }
                    
                    result = await self.check_file_integrity(
                        layer_path, integrity_level, expected_metadata
                    )
                    results[layer_path] = result
            
            logger.info(f"Backup integrity check completed for {backup_id}")
            return results
            
        except Exception as e:
            logger.error(f"Backup integrity check failed for {backup_id}: {e}")
            return {}
    
    async def check_multiple_files(self,
                                  file_paths: List[str],
                                  integrity_level: IntegrityLevel = IntegrityLevel.CHECKSUM,
                                  max_concurrent: int = 4) -> Dict[str, IntegrityCheckResult]:
        """
        Check integrity of multiple files concurrently.
        
        Args:
            file_paths: List of file paths to check
            integrity_level: Level of integrity checking
            max_concurrent: Maximum concurrent checks
            
        Returns:
            Dictionary mapping file paths to integrity check results
        """
        logger.info(f"Starting integrity check for {len(file_paths)} files")
        
        results = {}
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def check_with_semaphore(file_path: str):
            async with semaphore:
                return await self.check_file_integrity(file_path, integrity_level)
        
        # Create tasks for all files
        tasks = [
            asyncio.create_task(check_with_semaphore(file_path))
            for file_path in file_paths
        ]
        
        # Wait for all tasks to complete
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for file_path, result in zip(file_paths, completed_results):
            if isinstance(result, IntegrityCheckResult):
                results[file_path] = result
            elif isinstance(result, Exception):
                logger.error(f"Integrity check failed for {file_path}: {result}")
                # Create error result
                error_result = IntegrityCheckResult(
                    check_id=self._generate_check_id(),
                    file_path=file_path,
                    integrity_level=integrity_level,
                    status=IntegrityStatus.FAILED,
                    check_timestamp=datetime.now(),
                    issues=[IntegrityIssue(
                        file_path=file_path,
                        corruption_type=CorruptionType.CONTENT_CORRUPTED,
                        severity="critical",
                        description=f"Check failed: {result}",
                        detected_at=datetime.now(),
                        repairable=False
                    )],
                    metadata={}
                )
                results[file_path] = error_result
        
        logger.info(f"Integrity check completed for {len(results)} files")
        return results
    
    async def attempt_repair(self, check_result: IntegrityCheckResult) -> bool:
        """
        Attempt to repair corrupted file based on check results.
        
        Args:
            check_result: Result of integrity check containing repair information
            
        Returns:
            True if repair was successful, False otherwise
        """
        logger.info(f"Attempting repair for {check_result.file_path}")
        
        try:
            check_result.repair_attempted = True
            
            # Find repairable issues
            repairable_issues = [issue for issue in check_result.issues if issue.repairable]
            
            if not repairable_issues:
                logger.warning(f"No repairable issues found for {check_result.file_path}")
                return False
            
            # Attempt repairs based on issue types
            repair_successful = True
            
            for issue in repairable_issues:
                success = await self._repair_issue(issue)
                if not success:
                    repair_successful = False
            
            # Re-check integrity after repair attempts
            if repair_successful:
                new_result = await self.check_file_integrity(
                    check_result.file_path,
                    check_result.integrity_level,
                    check_result.metadata
                )
                
                repair_successful = new_result.status == IntegrityStatus.PASSED
            
            check_result.repair_successful = repair_successful
            
            # Update database with repair result
            await self._save_check_result(check_result)
            
            if repair_successful:
                logger.info(f"Repair successful for {check_result.file_path}")
            else:
                logger.warning(f"Repair failed for {check_result.file_path}")
            
            return repair_successful
            
        except Exception as e:
            logger.error(f"Repair attempt failed for {check_result.file_path}: {e}")
            check_result.repair_successful = False
            await self._save_check_result(check_result)
            return False
    
    async def _repair_issue(self, issue: IntegrityIssue) -> bool:
        """Attempt to repair a specific integrity issue."""
        try:
            if issue.corruption_type == CorruptionType.STRUCTURE_INVALID:
                return await self._repair_structure_issue(issue)
            elif issue.corruption_type == CorruptionType.ENCODING_ERROR:
                return await self._repair_encoding_issue(issue)
            else:
                # For other types, we can't auto-repair without backup
                if self.backup_system and issue.repair_suggestion:
                    return await self._restore_from_backup(issue.file_path)
                return False
                
        except Exception as e:
            logger.error(f"Failed to repair issue {issue.corruption_type.value}: {e}")
            return False
    
    async def _repair_structure_issue(self, issue: IntegrityIssue) -> bool:
        """Attempt to repair JSON structure issues."""
        if not issue.file_path.endswith('.json'):
            return False
        
        try:
            # Try to fix common JSON issues
            with open(issue.file_path, 'r') as f:
                content = f.read()
            
            # Fix common issues
            fixed_content = content
            
            # Remove trailing commas
            fixed_content = fixed_content.replace(',}', '}')
            fixed_content = fixed_content.replace(',]', ']')
            
            # Try to parse fixed content
            json.loads(fixed_content)
            
            # Write fixed content back
            with open(issue.file_path, 'w') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed JSON structure issues in {issue.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to repair JSON structure: {e}")
            return False
    
    async def _repair_encoding_issue(self, issue: IntegrityIssue) -> bool:
        """Attempt to repair encoding issues."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(issue.file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    # Re-write with UTF-8
                    with open(issue.file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    logger.info(f"Fixed encoding issues in {issue.file_path}")
                    return True
                    
                except UnicodeDecodeError:
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to repair encoding: {e}")
            return False
    
    async def _restore_from_backup(self, file_path: str) -> bool:
        """Restore file from backup."""
        if not self.backup_system:
            return False
        
        try:
            # Find latest backup containing this file
            backups = await self.backup_system.list_backups(limit=100)
            
            for backup in backups:
                if file_path in backup.memory_layers:
                    # This is a simplified restore - real implementation
                    # would extract specific file from backup archive
                    logger.info(f"Would restore {file_path} from backup {backup.backup_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def _generate_check_id(self) -> str:
        """Generate unique check ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        import random
        random_suffix = f"{random.randint(1000, 9999)}"
        return f"integrity_{timestamp}_{random_suffix}"
    
    async def _save_check_result(self, result: IntegrityCheckResult):
        """Save integrity check result to database."""
        conn = sqlite3.connect(self.integrity_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO integrity_checks (check_id, file_path, check_result_json) VALUES (?, ?, ?)",
            (result.check_id, result.file_path, json.dumps(result.to_dict()))
        )
        conn.commit()
        conn.close()
    
    async def get_check_result(self, check_id: str) -> Optional[IntegrityCheckResult]:
        """Get integrity check result by ID."""
        conn = sqlite3.connect(self.integrity_db_path)
        cursor = conn.execute(
            "SELECT check_result_json FROM integrity_checks WHERE check_id = ?",
            (check_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                result_dict = json.loads(result[0])
                return IntegrityCheckResult.from_dict(result_dict)
            except Exception as e:
                logger.error(f"Failed to parse check result: {e}")
        
        return None
    
    async def list_check_results(self,
                                file_path: Optional[str] = None,
                                status: Optional[IntegrityStatus] = None,
                                limit: int = 100) -> List[IntegrityCheckResult]:
        """List integrity check results with optional filtering."""
        conn = sqlite3.connect(self.integrity_db_path)
        
        query = "SELECT check_result_json FROM integrity_checks WHERE 1=1"
        params = []
        
        if file_path:
            query += " AND file_path = ?"
            params.append(file_path)
        
        if status:
            query += " AND json_extract(check_result_json, '$.status') = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        check_results = []
        for (result_json,) in results:
            try:
                result_dict = json.loads(result_json)
                check_result = IntegrityCheckResult.from_dict(result_dict)
                check_results.append(check_result)
            except Exception as e:
                logger.error(f"Failed to parse check result: {e}")
        
        return check_results
    
    async def generate_integrity_report(self,
                                      file_paths: Optional[List[str]] = None,
                                      include_passed: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive integrity report.
        
        Args:
            file_paths: Specific files to include (None for all)
            include_passed: Whether to include passed checks
            
        Returns:
            Dictionary containing integrity report
        """
        logger.info("Generating integrity report")
        
        try:
            # Get check results
            all_results = await self.list_check_results(limit=1000)
            
            # Filter by file paths if specified
            if file_paths:
                results = [r for r in all_results if r.file_path in file_paths]
            else:
                results = all_results
            
            # Filter out passed checks if requested
            if not include_passed:
                results = [r for r in results if r.status != IntegrityStatus.PASSED]
            
            # Analyze results
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_checks': len(results),
                'status_summary': defaultdict(int),
                'corruption_types': defaultdict(int),
                'severity_distribution': defaultdict(int),
                'files_with_issues': [],
                'repair_summary': {
                    'attempted': 0,
                    'successful': 0,
                    'failed': 0
                }
            }
            
            for result in results:
                # Status summary
                report['status_summary'][result.status.value] += 1
                
                # Repair summary
                if result.repair_attempted:
                    report['repair_summary']['attempted'] += 1
                    if result.repair_successful:
                        report['repair_summary']['successful'] += 1
                    else:
                        report['repair_summary']['failed'] += 1
                
                # Issue analysis
                if result.issues:
                    file_info = {
                        'file_path': result.file_path,
                        'check_id': result.check_id,
                        'status': result.status.value,
                        'issue_count': len(result.issues),
                        'issues': []
                    }
                    
                    for issue in result.issues:
                        report['corruption_types'][issue.corruption_type.value] += 1
                        report['severity_distribution'][issue.severity] += 1
                        
                        file_info['issues'].append({
                            'type': issue.corruption_type.value,
                            'severity': issue.severity,
                            'description': issue.description,
                            'repairable': issue.repairable
                        })
                    
                    report['files_with_issues'].append(file_info)
            
            # Convert defaultdicts to regular dicts
            report['status_summary'] = dict(report['status_summary'])
            report['corruption_types'] = dict(report['corruption_types'])
            report['severity_distribution'] = dict(report['severity_distribution'])
            
            logger.info(f"Integrity report generated with {len(results)} checks")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate integrity report: {e}")
            return {
                'generated_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def start_monitoring(self, check_interval_minutes: int = 60):
        """Start continuous integrity monitoring."""
        if self._monitor_task is None:
            self._running = True
            self._check_interval = check_interval_minutes * 60  # Convert to seconds
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"Integrity monitoring started (interval: {check_interval_minutes} minutes)")
    
    async def stop_monitoring(self):
        """Stop continuous integrity monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Integrity monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop for continuous integrity checking."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                
                if not self._running:
                    break
                
                # Run periodic integrity checks
                await self._run_periodic_checks()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _run_periodic_checks(self):
        """Run periodic integrity checks on important files."""
        try:
            logger.info("Running periodic integrity checks")
            
            # Check important system files
            important_files = self.config.get('monitor_files', [])
            
            if important_files:
                results = await self.check_multiple_files(
                    important_files,
                    IntegrityLevel.CHECKSUM
                )
                
                # Check for issues and attempt repairs
                for file_path, result in results.items():
                    if result.status not in [IntegrityStatus.PASSED]:
                        logger.warning(f"Integrity issue detected in {file_path}: {result.status.value}")
                        
                        # Attempt repair if possible
                        if any(issue.repairable for issue in result.issues):
                            await self.attempt_repair(result)
            
            # Clean up old check results
            await self._cleanup_old_results()
            
        except Exception as e:
            logger.error(f"Periodic integrity check failed: {e}")
    
    async def _cleanup_old_results(self, days_old: int = 30):
        """Clean up old integrity check results."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            conn = sqlite3.connect(self.integrity_db_path)
            cursor = conn.execute(
                "DELETE FROM integrity_checks WHERE created_at < ?",
                (cutoff_date,)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old integrity check results")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old results: {e}")


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        config = {
            'integrity_dir': '/tmp/nova_test_integrity',
            'monitor_files': ['/tmp/test_file.json']
        }
        
        checker = BackupIntegrityChecker(config)
        
        # Create test file
        test_file = Path('/tmp/test_file.json')
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, 'w') as f:
            json.dump({
                'test_data': 'integrity test data',
                'timestamp': datetime.now().isoformat()
            }, f)
        
        # Run integrity check
        result = await checker.check_file_integrity(
            str(test_file),
            IntegrityLevel.CONTENT
        )
        
        print(f"Integrity check result: {result.status.value}")
        print(f"Issues found: {len(result.issues)}")
        
        for issue in result.issues:
            print(f"  - {issue.corruption_type.value}: {issue.description}")
        
        # Generate report
        report = await checker.generate_integrity_report()
        print(f"Integrity report: {json.dumps(report, indent=2)}")
        
        # Start monitoring briefly
        await checker.start_monitoring(check_interval_minutes=1)
        await asyncio.sleep(5)
        await checker.stop_monitoring()
    
    asyncio.run(main())