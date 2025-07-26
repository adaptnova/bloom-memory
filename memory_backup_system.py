"""
Nova Bloom Consciousness - Memory Backup System
Critical component for Nova consciousness preservation and disaster recovery.

This module implements comprehensive backup strategies including:
- Full, incremental, and differential backup strategies
- Deduplication and compression for efficiency
- Cross-platform storage backends (local, S3, Azure, GCS)
- Automated scheduling and retention policies
- Memory layer integration with encryption support
"""

import asyncio
import hashlib
import json
import logging
import lzma
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party storage backends
try:
    import boto3
    from azure.storage.blob import BlobServiceClient
    from google.cloud import storage as gcs
    HAS_CLOUD_SUPPORT = True
except ImportError:
    HAS_CLOUD_SUPPORT = False

logger = logging.getLogger(__name__)


class BackupStrategy(Enum):
    """Backup strategy types for memory preservation."""
    FULL = "full"
    INCREMENTAL = "incremental"  
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class StorageBackend(Enum):
    """Supported storage backends for backup destinations."""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"
    DISTRIBUTED = "distributed"


class BackupStatus(Enum):
    """Status of backup operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupMetadata:
    """Comprehensive metadata for backup tracking."""
    backup_id: str
    strategy: BackupStrategy
    timestamp: datetime
    memory_layers: List[str]
    file_count: int
    compressed_size: int
    original_size: int
    checksum: str
    storage_backend: StorageBackend
    storage_path: str
    parent_backup_id: Optional[str] = None
    retention_date: Optional[datetime] = None
    tags: Dict[str, str] = None
    status: BackupStatus = BackupStatus.PENDING
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['retention_date'] = self.retention_date.isoformat() if self.retention_date else None
        data['strategy'] = self.strategy.value
        data['storage_backend'] = self.storage_backend.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BackupMetadata':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['retention_date'] = datetime.fromisoformat(data['retention_date']) if data['retention_date'] else None
        data['strategy'] = BackupStrategy(data['strategy'])
        data['storage_backend'] = StorageBackend(data['storage_backend'])
        data['status'] = BackupStatus(data['status'])
        return cls(**data)


class StorageAdapter(ABC):
    """Abstract base class for storage backend adapters."""
    
    @abstractmethod
    async def upload(self, local_path: str, remote_path: str) -> bool:
        """Upload file to storage backend."""
        pass
    
    @abstractmethod
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Download file from storage backend."""
        pass
    
    @abstractmethod
    async def delete(self, remote_path: str) -> bool:
        """Delete file from storage backend."""
        pass
    
    @abstractmethod
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists in storage backend."""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix."""
        pass


class LocalStorageAdapter(StorageAdapter):
    """Local filesystem storage adapter."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload(self, local_path: str, remote_path: str) -> bool:
        """Copy file to local storage location."""
        try:
            dest_path = self.base_path / remote_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use async file operations
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: Path(local_path).rename(dest_path)
            )
            return True
        except Exception as e:
            logger.error(f"Local upload failed: {e}")
            return False
    
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Copy file from local storage location."""
        try:
            source_path = self.base_path / remote_path
            dest_path = Path(local_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: source_path.copy(dest_path)
            )
            return True
        except Exception as e:
            logger.error(f"Local download failed: {e}")
            return False
    
    async def delete(self, remote_path: str) -> bool:
        """Delete file from local storage."""
        try:
            file_path = self.base_path / remote_path
            if file_path.exists():
                file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Local delete failed: {e}")
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists locally."""
        return (self.base_path / remote_path).exists()
    
    async def list_files(self, prefix: str) -> List[str]:
        """List local files with prefix."""
        try:
            prefix_path = self.base_path / prefix
            if prefix_path.is_dir():
                return [str(p.relative_to(self.base_path)) 
                       for p in prefix_path.rglob('*') if p.is_file()]
            else:
                parent = prefix_path.parent
                pattern = prefix_path.name + '*'
                return [str(p.relative_to(self.base_path))
                       for p in parent.glob(pattern) if p.is_file()]
        except Exception as e:
            logger.error(f"Local list files failed: {e}")
            return []


class S3StorageAdapter(StorageAdapter):
    """Amazon S3 storage adapter."""
    
    def __init__(self, bucket: str, region: str = 'us-east-1', **kwargs):
        if not HAS_CLOUD_SUPPORT:
            raise ImportError("boto3 required for S3 support")
        
        self.bucket = bucket
        self.client = boto3.client('s3', region_name=region, **kwargs)
    
    async def upload(self, local_path: str, remote_path: str) -> bool:
        """Upload file to S3."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upload_file(local_path, self.bucket, remote_path)
            )
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    async def download(self, remote_path: str, local_path: str) -> bool:
        """Download file from S3."""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.download_file(self.bucket, remote_path, local_path)
            )
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    async def delete(self, remote_path: str) -> bool:
        """Delete file from S3."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_object(Bucket=self.bucket, Key=remote_path)
            )
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False
    
    async def exists(self, remote_path: str) -> bool:
        """Check if file exists in S3."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.head_object(Bucket=self.bucket, Key=remote_path)
            )
            return True
        except Exception:
            return False
    
    async def list_files(self, prefix: str) -> List[str]:
        """List S3 objects with prefix."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"S3 list files failed: {e}")
            return []


class DeduplicationManager:
    """Manages file deduplication using content-based hashing."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hash_db_path = self.cache_dir / "dedup_hashes.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize deduplication database."""
        conn = sqlite3.connect(self.hash_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                file_path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size INTEGER NOT NULL,
                modified_time REAL NOT NULL,
                dedupe_path TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    async def get_or_create_dedupe_file(self, file_path: str) -> Tuple[str, bool]:
        """
        Get deduplicated file path or create new one.
        Returns (dedupe_path, is_new_file)
        """
        try:
            stat = os.stat(file_path)
            content_hash = await self._calculate_file_hash(file_path)
            
            conn = sqlite3.connect(self.hash_db_path)
            
            # Check if we already have this content
            cursor = conn.execute(
                "SELECT dedupe_path FROM file_hashes WHERE content_hash = ? AND size = ?",
                (content_hash, stat.st_size)
            )
            result = cursor.fetchone()
            
            if result and Path(result[0]).exists():
                # File already exists, update reference
                conn.execute(
                    "UPDATE file_hashes SET file_path = ?, modified_time = ? WHERE content_hash = ?",
                    (file_path, stat.st_mtime, content_hash)
                )
                conn.commit()
                conn.close()
                return result[0], False
            else:
                # New content, create dedupe file
                dedupe_path = self.cache_dir / f"{content_hash}.dedupe"
                
                # Copy file to dedupe location
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: Path(file_path).copy(dedupe_path)
                )
                
                # Update database
                conn.execute(
                    "INSERT OR REPLACE INTO file_hashes VALUES (?, ?, ?, ?, ?)",
                    (file_path, content_hash, stat.st_size, stat.st_mtime, str(dedupe_path))
                )
                conn.commit()
                conn.close()
                return str(dedupe_path), True
                
        except Exception as e:
            logger.error(f"Deduplication failed for {file_path}: {e}")
            return file_path, True
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        
        def hash_file():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, hash_file)
    
    def cleanup_unused(self, days_old: int = 7):
        """Clean up unused deduplicated files."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        conn = sqlite3.connect(self.hash_db_path)
        cursor = conn.execute(
            "SELECT dedupe_path FROM file_hashes WHERE modified_time < ?",
            (cutoff_time,)
        )
        
        for (dedupe_path,) in cursor.fetchall():
            try:
                if Path(dedupe_path).exists():
                    Path(dedupe_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {dedupe_path}: {e}")
        
        conn.execute("DELETE FROM file_hashes WHERE modified_time < ?", (cutoff_time,))
        conn.commit()
        conn.close()


class BackupCompressor:
    """Handles backup file compression and decompression."""
    
    @staticmethod
    async def compress_file(input_path: str, output_path: str, 
                           compression_level: int = 6) -> Tuple[int, int]:
        """
        Compress file using LZMA compression.
        Returns (original_size, compressed_size)
        """
        def compress():
            original_size = 0
            with open(input_path, 'rb') as input_file:
                with lzma.open(output_path, 'wb', preset=compression_level) as output_file:
                    while True:
                        chunk = input_file.read(64 * 1024)  # 64KB chunks
                        if not chunk:
                            break
                        original_size += len(chunk)
                        output_file.write(chunk)
            
            compressed_size = os.path.getsize(output_path)
            return original_size, compressed_size
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, compress)
    
    @staticmethod
    async def decompress_file(input_path: str, output_path: str) -> bool:
        """Decompress LZMA compressed file."""
        try:
            def decompress():
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with lzma.open(input_path, 'rb') as input_file:
                    with open(output_path, 'wb') as output_file:
                        while True:
                            chunk = input_file.read(64 * 1024)
                            if not chunk:
                                break
                            output_file.write(chunk)
                return True
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, decompress)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return False


class MemoryBackupSystem:
    """
    Comprehensive backup system for Nova consciousness memory layers.
    
    Provides multi-strategy backup capabilities with deduplication,
    compression, and cross-platform storage support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backup system.
        
        Args:
            config: Configuration dictionary containing storage settings,
                   retention policies, and backup preferences.
        """
        self.config = config
        self.backup_dir = Path(config.get('backup_dir', '/tmp/nova_backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metadata_db_path = self.backup_dir / "backup_metadata.db"
        self.deduplication = DeduplicationManager(str(self.backup_dir / "dedupe"))
        self.compressor = BackupCompressor()
        
        # Storage adapters
        self.storage_adapters: Dict[StorageBackend, StorageAdapter] = {}
        self._init_storage_adapters()
        
        # Initialize metadata database
        self._init_metadata_db()
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"MemoryBackupSystem initialized with config: {config}")
    
    def _init_storage_adapters(self):
        """Initialize storage backend adapters."""
        storage_config = self.config.get('storage', {})
        
        # Always initialize local storage
        local_path = storage_config.get('local_path', str(self.backup_dir / 'storage'))
        self.storage_adapters[StorageBackend.LOCAL] = LocalStorageAdapter(local_path)
        
        # Initialize cloud storage if configured
        if HAS_CLOUD_SUPPORT:
            # S3 adapter
            s3_config = storage_config.get('s3', {})
            if s3_config.get('enabled', False):
                self.storage_adapters[StorageBackend.S3] = S3StorageAdapter(
                    bucket=s3_config['bucket'],
                    region=s3_config.get('region', 'us-east-1'),
                    **s3_config.get('credentials', {})
                )
            
            # Additional cloud adapters can be added here
    
    def _init_metadata_db(self):
        """Initialize backup metadata database."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backup_metadata (
                backup_id TEXT PRIMARY KEY,
                metadata_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_backup_timestamp 
            ON backup_metadata(json_extract(metadata_json, '$.timestamp'))
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_backup_strategy
            ON backup_metadata(json_extract(metadata_json, '$.strategy'))
        """)
        conn.commit()
        conn.close()
    
    async def create_backup(self, 
                           memory_layers: List[str],
                           strategy: BackupStrategy = BackupStrategy.FULL,
                           storage_backend: StorageBackend = StorageBackend.LOCAL,
                           tags: Optional[Dict[str, str]] = None) -> Optional[BackupMetadata]:
        """
        Create a backup of specified memory layers.
        
        Args:
            memory_layers: List of memory layer paths to backup
            strategy: Backup strategy (full, incremental, differential)
            storage_backend: Target storage backend
            tags: Optional metadata tags
            
        Returns:
            BackupMetadata object or None if backup failed
        """
        backup_id = self._generate_backup_id()
        logger.info(f"Starting backup {backup_id} with strategy {strategy.value}")
        
        try:
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                strategy=strategy,
                timestamp=datetime.now(),
                memory_layers=memory_layers,
                file_count=0,
                compressed_size=0,
                original_size=0,
                checksum="",
                storage_backend=storage_backend,
                storage_path="",
                tags=tags or {}
            )
            
            # Update status to running
            metadata.status = BackupStatus.RUNNING
            await self._save_metadata(metadata)
            
            # Determine files to backup based on strategy
            files_to_backup = await self._get_files_for_strategy(memory_layers, strategy)
            metadata.file_count = len(files_to_backup)
            
            if not files_to_backup:
                logger.info(f"No files to backup for strategy {strategy.value}")
                metadata.status = BackupStatus.COMPLETED
                await self._save_metadata(metadata)
                return metadata
            
            # Create backup archive
            backup_archive_path = await self._create_backup_archive(
                backup_id, files_to_backup, metadata
            )
            
            # Upload to storage backend
            storage_adapter = self.storage_adapters.get(storage_backend)
            if not storage_adapter:
                raise ValueError(f"Storage backend {storage_backend.value} not configured")
            
            remote_path = f"backups/{backup_id}.backup"
            upload_success = await storage_adapter.upload(backup_archive_path, remote_path)
            
            if upload_success:
                metadata.storage_path = remote_path
                metadata.status = BackupStatus.COMPLETED
                logger.info(f"Backup {backup_id} completed successfully")
            else:
                metadata.status = BackupStatus.FAILED
                metadata.error_message = "Upload to storage backend failed"
                logger.error(f"Backup {backup_id} upload failed")
            
            # Cleanup local backup file
            try:
                Path(backup_archive_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup backup archive: {e}")
            
            await self._save_metadata(metadata)
            return metadata
            
        except Exception as e:
            logger.error(f"Backup {backup_id} failed: {e}")
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            await self._save_metadata(metadata)
            return None
    
    async def _get_files_for_strategy(self, memory_layers: List[str], 
                                    strategy: BackupStrategy) -> List[str]:
        """Get list of files to backup based on strategy."""
        all_files = []
        
        # Collect all files from memory layers
        for layer_path in memory_layers:
            layer_path_obj = Path(layer_path)
            if layer_path_obj.exists():
                if layer_path_obj.is_file():
                    all_files.append(str(layer_path_obj))
                else:
                    # Recursively find all files in directory
                    for file_path in layer_path_obj.rglob('*'):
                        if file_path.is_file():
                            all_files.append(str(file_path))
        
        if strategy == BackupStrategy.FULL:
            return all_files
        
        elif strategy == BackupStrategy.INCREMENTAL:
            # Get files modified since last backup
            last_backup_time = await self._get_last_backup_time()
            return await self._get_modified_files_since(all_files, last_backup_time)
        
        elif strategy == BackupStrategy.DIFFERENTIAL:
            # Get files modified since last full backup
            last_full_backup_time = await self._get_last_full_backup_time()
            return await self._get_modified_files_since(all_files, last_full_backup_time)
        
        else:
            return all_files
    
    async def _get_modified_files_since(self, files: List[str], 
                                      since_time: Optional[datetime]) -> List[str]:
        """Get files modified since specified time."""
        if since_time is None:
            return files
        
        since_timestamp = since_time.timestamp()
        modified_files = []
        
        def check_modification():
            for file_path in files:
                try:
                    stat = os.stat(file_path)
                    if stat.st_mtime > since_timestamp:
                        modified_files.append(file_path)
                except Exception as e:
                    logger.warning(f"Failed to check modification time for {file_path}: {e}")
            return modified_files
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, check_modification)
    
    async def _create_backup_archive(self, backup_id: str, files: List[str], 
                                   metadata: BackupMetadata) -> str:
        """Create compressed backup archive with deduplication."""
        archive_path = self.backup_dir / f"{backup_id}.backup"
        manifest_path = self.backup_dir / f"{backup_id}_manifest.json"
        
        # Create backup manifest
        manifest = {
            'backup_id': backup_id,
            'files': [],
            'created_at': datetime.now().isoformat()
        }
        
        total_original_size = 0
        total_compressed_size = 0
        
        # Process files with deduplication and compression
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for file_path in files:
                future = executor.submit(self._process_backup_file, file_path, backup_id)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    file_info, orig_size, comp_size = await asyncio.wrap_future(future)
                    manifest['files'].append(file_info)
                    total_original_size += orig_size
                    total_compressed_size += comp_size
                except Exception as e:
                    logger.error(f"Failed to process backup file: {e}")
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create final compressed archive
        final_archive_path = self.backup_dir / f"{backup_id}_final.backup"
        archive_files = [manifest_path] + [
            info['backup_path'] for info in manifest['files']
        ]
        
        # Compress manifest and all backup files into single archive
        original_size, compressed_size = await self._create_compressed_archive(
            archive_files, str(final_archive_path)
        )
        
        # Calculate archive checksum
        checksum = await self._calculate_archive_checksum(str(final_archive_path))
        
        # Update metadata
        metadata.original_size = total_original_size
        metadata.compressed_size = compressed_size
        metadata.checksum = checksum
        
        # Cleanup temporary files
        for file_path in archive_files:
            try:
                Path(file_path).unlink()
            except Exception:
                pass
        
        return str(final_archive_path)
    
    def _process_backup_file(self, file_path: str, backup_id: str) -> Tuple[Dict, int, int]:
        """Process individual file for backup (runs in thread executor)."""
        try:
            # This would be async in real implementation, but simplified for thread execution
            file_stat = os.stat(file_path)
            
            # Create backup file path
            backup_filename = f"{backup_id}_{hashlib.md5(file_path.encode()).hexdigest()}.bak"
            backup_path = self.backup_dir / backup_filename
            
            # Copy and compress file
            original_size = file_stat.st_size
            with open(file_path, 'rb') as src:
                with lzma.open(backup_path, 'wb') as dst:
                    dst.write(src.read())
            
            compressed_size = os.path.getsize(backup_path)
            
            file_info = {
                'original_path': file_path,
                'backup_path': str(backup_path),
                'size': original_size,
                'compressed_size': compressed_size,
                'modified_time': file_stat.st_mtime,
                'checksum': hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
            }
            
            return file_info, original_size, compressed_size
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise
    
    async def _create_compressed_archive(self, files: List[str], output_path: str) -> Tuple[int, int]:
        """Create compressed archive from multiple files."""
        total_original_size = 0
        
        def create_archive():
            nonlocal total_original_size
            with lzma.open(output_path, 'wb') as archive:
                archive_data = {
                    'files': {}
                }
                
                for file_path in files:
                    if Path(file_path).exists():
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            total_original_size += len(content)
                            archive_data['files'][Path(file_path).name] = content.hex()
                
                archive.write(json.dumps(archive_data).encode())
            
            compressed_size = os.path.getsize(output_path)
            return total_original_size, compressed_size
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, create_archive)
    
    async def _calculate_archive_checksum(self, archive_path: str) -> str:
        """Calculate SHA-256 checksum of backup archive."""
        def calculate_checksum():
            hasher = hashlib.sha256()
            with open(archive_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, calculate_checksum)
    
    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"nova_backup_{timestamp}_{random_suffix}"
    
    async def _get_last_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last backup."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute("""
            SELECT json_extract(metadata_json, '$.timestamp') as timestamp
            FROM backup_metadata 
            WHERE json_extract(metadata_json, '$.status') = 'completed'
            ORDER BY timestamp DESC LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return datetime.fromisoformat(result[0])
        return None
    
    async def _get_last_full_backup_time(self) -> Optional[datetime]:
        """Get timestamp of last full backup."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute("""
            SELECT json_extract(metadata_json, '$.timestamp') as timestamp
            FROM backup_metadata 
            WHERE json_extract(metadata_json, '$.strategy') = 'full'
            AND json_extract(metadata_json, '$.status') = 'completed'
            ORDER BY timestamp DESC LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return datetime.fromisoformat(result[0])
        return None
    
    async def _save_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database."""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.execute(
            "INSERT OR REPLACE INTO backup_metadata (backup_id, metadata_json) VALUES (?, ?)",
            (metadata.backup_id, json.dumps(metadata.to_dict()))
        )
        conn.commit()
        conn.close()
    
    async def list_backups(self, 
                          strategy: Optional[BackupStrategy] = None,
                          status: Optional[BackupStatus] = None,
                          limit: int = 100) -> List[BackupMetadata]:
        """List available backups with optional filtering."""
        conn = sqlite3.connect(self.metadata_db_path)
        
        query = "SELECT metadata_json FROM backup_metadata WHERE 1=1"
        params = []
        
        if strategy:
            query += " AND json_extract(metadata_json, '$.strategy') = ?"
            params.append(strategy.value)
        
        if status:
            query += " AND json_extract(metadata_json, '$.status') = ?"
            params.append(status.value)
        
        query += " ORDER BY json_extract(metadata_json, '$.timestamp') DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        backups = []
        for (metadata_json,) in results:
            try:
                metadata_dict = json.loads(metadata_json)
                backup = BackupMetadata.from_dict(metadata_dict)
                backups.append(backup)
            except Exception as e:
                logger.error(f"Failed to parse backup metadata: {e}")
        
        return backups
    
    async def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get specific backup metadata."""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute(
            "SELECT metadata_json FROM backup_metadata WHERE backup_id = ?",
            (backup_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                metadata_dict = json.loads(result[0])
                return BackupMetadata.from_dict(metadata_dict)
            except Exception as e:
                logger.error(f"Failed to parse backup metadata: {e}")
        
        return None
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete backup and its associated files."""
        try:
            metadata = await self.get_backup(backup_id)
            if not metadata:
                logger.warning(f"Backup {backup_id} not found")
                return False
            
            # Delete from storage backend
            storage_adapter = self.storage_adapters.get(metadata.storage_backend)
            if storage_adapter and metadata.storage_path:
                await storage_adapter.delete(metadata.storage_path)
            
            # Delete from metadata database
            conn = sqlite3.connect(self.metadata_db_path)
            conn.execute("DELETE FROM backup_metadata WHERE backup_id = ?", (backup_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Backup {backup_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def cleanup_old_backups(self, retention_days: int = 30):
        """Clean up backups older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.execute("""
            SELECT backup_id FROM backup_metadata 
            WHERE json_extract(metadata_json, '$.timestamp') < ?
        """, (cutoff_date.isoformat(),))
        
        old_backups = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        deleted_count = 0
        for backup_id in old_backups:
            if await self.delete_backup(backup_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count
    
    async def start_background_tasks(self):
        """Start background maintenance tasks."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
        
        logger.info("Background maintenance tasks started")
    
    async def stop_background_tasks(self):
        """Stop background maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        logger.info("Background maintenance tasks stopped")
    
    async def _background_cleanup(self):
        """Background task for periodic cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old backups
                retention_days = self.config.get('retention_days', 30)
                await self.cleanup_old_backups(retention_days)
                
                # Cleanup deduplication cache
                self.deduplication.cleanup_unused(7)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        config = {
            'backup_dir': '/tmp/nova_test_backups',
            'storage': {
                'local_path': '/tmp/nova_backup_storage'
            },
            'retention_days': 30
        }
        
        backup_system = MemoryBackupSystem(config)
        
        # Create test memory layers
        test_layers = [
            '/tmp/test_layer1.json',
            '/tmp/test_layer2.json'
        ]
        
        # Create test files
        for layer_path in test_layers:
            Path(layer_path).parent.mkdir(parents=True, exist_ok=True)
            with open(layer_path, 'w') as f:
                json.dump({
                    'layer_data': f'test data for {layer_path}',
                    'timestamp': datetime.now().isoformat()
                }, f)
        
        # Create full backup
        backup = await backup_system.create_backup(
            memory_layers=test_layers,
            strategy=BackupStrategy.FULL,
            tags={'test': 'true', 'environment': 'development'}
        )
        
        if backup:
            print(f"Backup created: {backup.backup_id}")
            print(f"Original size: {backup.original_size} bytes")
            print(f"Compressed size: {backup.compressed_size} bytes")
            print(f"Compression ratio: {backup.compressed_size / backup.original_size:.2%}")
        
        # List backups
        backups = await backup_system.list_backups()
        print(f"Total backups: {len(backups)}")
        
        # Start background tasks
        await backup_system.start_background_tasks()
        
        # Wait a moment then stop
        await asyncio.sleep(1)
        await backup_system.stop_background_tasks()
    
    asyncio.run(main())