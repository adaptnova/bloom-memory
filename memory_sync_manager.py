#!/usr/bin/env python3
"""
Memory Sync Manager
Real-time synchronization manager for Nova memory systems
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import weakref

from cross_nova_transfer_protocol import (
    CrossNovaTransferProtocol, TransferOperation, TransferStatus,
    VectorClock, MemoryDelta, ConflictResolution, ConflictResolver
)
from unified_memory_api import NovaMemoryAPI, MemoryRequest, MemoryResponse, MemoryOperation

logger = logging.getLogger(__name__)

class SyncMode(Enum):
    """Synchronization modes"""
    FULL = "full"
    INCREMENTAL = "incremental"
    SELECTIVE = "selective"
    REAL_TIME = "real_time"
    BACKUP_ONLY = "backup_only"

class SyncDirection(Enum):
    """Synchronization directions"""
    BIDIRECTIONAL = "bidirectional"
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    BROADCAST = "broadcast"

class SyncStatus(Enum):
    """Synchronization status"""
    IDLE = "idle"
    SYNCING = "syncing"
    MONITORING = "monitoring"
    PAUSED = "paused"
    ERROR = "error"

class PrivacyLevel(Enum):
    """Memory privacy levels"""
    PUBLIC = "public"
    TEAM = "team"
    PRIVATE = "private"
    CLASSIFIED = "classified"

@dataclass
class SyncConfiguration:
    """Synchronization configuration"""
    target_nova: str
    target_host: str
    target_port: int
    sync_mode: SyncMode = SyncMode.INCREMENTAL
    sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    sync_interval: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    memory_types: List[str] = field(default_factory=list)
    privacy_levels: List[PrivacyLevel] = field(default_factory=lambda: [PrivacyLevel.PUBLIC, PrivacyLevel.TEAM])
    conflict_resolution: ConflictResolution = ConflictResolution.LATEST_WINS
    bandwidth_limit: int = 5 * 1024 * 1024  # 5MB/s
    compression_enabled: bool = True
    encryption_enabled: bool = True
    max_memory_age: Optional[timedelta] = None
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

@dataclass
class SyncSession:
    """Active synchronization session"""
    session_id: str
    config: SyncConfiguration
    status: SyncStatus = SyncStatus.IDLE
    started_at: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    next_sync: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'target_nova': self.config.target_nova,
            'sync_mode': self.config.sync_mode.value,
            'sync_direction': self.config.sync_direction.value,
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'next_sync': self.next_sync.isoformat() if self.next_sync else None,
            'errors': self.errors[-10:],  # Last 10 errors
            'stats': self.stats
        }

@dataclass
class MemorySnapshot:
    """Snapshot of memory state for sync comparison"""
    nova_id: str
    timestamp: datetime
    memory_checksums: Dict[str, str]
    total_count: int
    last_modified: Dict[str, datetime]
    vector_clock: VectorClock
    
    def calculate_deltas(self, other: 'MemorySnapshot') -> List[MemoryDelta]:
        """Calculate deltas between two snapshots"""
        deltas = []
        
        # Find new/modified memories
        for memory_id, checksum in self.memory_checksums.items():
            other_checksum = other.memory_checksums.get(memory_id)
            
            if other_checksum is None:
                # New memory
                delta = MemoryDelta(
                    memory_id=memory_id,
                    operation='create',
                    timestamp=self.last_modified.get(memory_id, self.timestamp),
                    vector_clock=self.vector_clock
                )
                delta.calculate_checksum()
                deltas.append(delta)
                
            elif other_checksum != checksum:
                # Modified memory
                delta = MemoryDelta(
                    memory_id=memory_id,
                    operation='update',
                    timestamp=self.last_modified.get(memory_id, self.timestamp),
                    vector_clock=self.vector_clock
                )
                delta.calculate_checksum()
                deltas.append(delta)
        
        # Find deleted memories
        for memory_id in other.memory_checksums:
            if memory_id not in self.memory_checksums:
                delta = MemoryDelta(
                    memory_id=memory_id,
                    operation='delete',
                    timestamp=self.timestamp,
                    vector_clock=self.vector_clock
                )
                delta.calculate_checksum()
                deltas.append(delta)
        
        return deltas

class PrivacyController:
    """Controls what memories can be shared based on privacy settings"""
    
    def __init__(self):
        self.privacy_rules: Dict[str, Dict[str, Any]] = {}
        self.team_memberships: Dict[str, Set[str]] = {}
        self.classification_levels: Dict[str, int] = {
            PrivacyLevel.PUBLIC.value: 0,
            PrivacyLevel.TEAM.value: 1,
            PrivacyLevel.PRIVATE.value: 2,
            PrivacyLevel.CLASSIFIED.value: 3
        }
    
    def set_privacy_rule(self, memory_pattern: str, privacy_level: PrivacyLevel, 
                        allowed_novas: Optional[Set[str]] = None):
        """Set privacy rule for memory pattern"""
        self.privacy_rules[memory_pattern] = {
            'privacy_level': privacy_level,
            'allowed_novas': allowed_novas or set(),
            'created_at': datetime.now()
        }
    
    def add_team_membership(self, team_name: str, nova_ids: Set[str]):
        """Add team membership"""
        self.team_memberships[team_name] = nova_ids
    
    def can_share_memory(self, memory: Dict[str, Any], target_nova: str, 
                        source_nova: str) -> bool:
        """Check if memory can be shared with target Nova"""
        memory_id = memory.get('id', '')
        memory_content = str(memory.get('content', ''))
        memory_tags = memory.get('tags', [])
        
        # Get privacy level from memory or apply default rules
        privacy_level = self._determine_privacy_level(memory, memory_id, memory_content, memory_tags)
        
        if privacy_level == PrivacyLevel.PUBLIC:
            return True
        elif privacy_level == PrivacyLevel.PRIVATE:
            return target_nova == source_nova
        elif privacy_level == PrivacyLevel.CLASSIFIED:
            return False
        elif privacy_level == PrivacyLevel.TEAM:
            # Check team membership
            for team_novas in self.team_memberships.values():
                if source_nova in team_novas and target_nova in team_novas:
                    return True
            return False
        
        return False
    
    def _determine_privacy_level(self, memory: Dict[str, Any], memory_id: str, 
                               content: str, tags: List[str]) -> PrivacyLevel:
        """Determine privacy level for a memory"""
        # Check explicit privacy level
        if 'privacy_level' in memory:
            return PrivacyLevel(memory['privacy_level'])
        
        # Check patterns against rules
        for pattern, rule in self.privacy_rules.items():
            if (pattern in memory_id or pattern in content or 
                any(pattern in tag for tag in tags)):
                return rule['privacy_level']
        
        # Check tags for privacy indicators
        if any(tag in ['private', 'personal', 'confidential'] for tag in tags):
            return PrivacyLevel.PRIVATE
        elif any(tag in ['classified', 'secret', 'restricted'] for tag in tags):
            return PrivacyLevel.CLASSIFIED
        elif any(tag in ['team', 'internal', 'group'] for tag in tags):
            return PrivacyLevel.TEAM
        
        # Default to public
        return PrivacyLevel.PUBLIC

class BandwidthOptimizer:
    """Optimizes bandwidth usage during synchronization"""
    
    def __init__(self):
        self.transfer_stats: Dict[str, Dict[str, Any]] = {}
        self.network_conditions: Dict[str, float] = {}
    
    def record_transfer_stats(self, target_nova: str, bytes_transferred: int, 
                            duration: float, compression_ratio: float):
        """Record transfer statistics"""
        if target_nova not in self.transfer_stats:
            self.transfer_stats[target_nova] = {
                'total_bytes': 0,
                'total_duration': 0,
                'transfer_count': 0,
                'avg_compression_ratio': 0,
                'avg_throughput': 0
            }
        
        stats = self.transfer_stats[target_nova]
        stats['total_bytes'] += bytes_transferred
        stats['total_duration'] += duration
        stats['transfer_count'] += 1
        stats['avg_compression_ratio'] = (
            (stats['avg_compression_ratio'] * (stats['transfer_count'] - 1) + compression_ratio) /
            stats['transfer_count']
        )
        stats['avg_throughput'] = stats['total_bytes'] / stats['total_duration'] if stats['total_duration'] > 0 else 0
    
    def get_optimal_chunk_size(self, target_nova: str) -> int:
        """Get optimal chunk size based on network conditions"""
        base_chunk_size = 1024 * 1024  # 1MB
        
        if target_nova not in self.transfer_stats:
            return base_chunk_size
        
        stats = self.transfer_stats[target_nova]
        throughput = stats['avg_throughput']
        
        # Adjust chunk size based on throughput
        if throughput < 1024 * 1024:  # < 1MB/s
            return base_chunk_size // 4  # 256KB
        elif throughput > 10 * 1024 * 1024:  # > 10MB/s
            return base_chunk_size * 4  # 4MB
        else:
            return base_chunk_size
    
    def should_enable_compression(self, target_nova: str, data_size: int) -> bool:
        """Determine if compression should be enabled"""
        if target_nova not in self.transfer_stats:
            return data_size > 1024  # Enable for data > 1KB
        
        stats = self.transfer_stats[target_nova]
        compression_ratio = stats['avg_compression_ratio']
        throughput = stats['avg_throughput']
        
        # If compression ratio is poor or network is very fast, skip compression
        if compression_ratio < 1.2 and throughput > 50 * 1024 * 1024:  # 50MB/s
            return False
        
        return data_size > 512  # Enable for data > 512B

class MemorySyncManager:
    """Main memory synchronization manager"""
    
    def __init__(self, nova_id: str, memory_api: NovaMemoryAPI):
        self.nova_id = nova_id
        self.memory_api = memory_api
        self.transfer_protocol = CrossNovaTransferProtocol(nova_id)
        self.privacy_controller = PrivacyController()
        self.bandwidth_optimizer = BandwidthOptimizer()
        self.conflict_resolver = ConflictResolver()
        
        self.active_sessions: Dict[str, SyncSession] = {}
        self.snapshots: Dict[str, MemorySnapshot] = {}
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Weak references to avoid circular dependencies
        self.sync_callbacks: List[weakref.WeakMethod] = []
    
    async def start(self):
        """Start the sync manager"""
        await self.transfer_protocol.start_server()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.is_running = True
        logger.info(f"Memory Sync Manager started for Nova {self.nova_id}")
    
    async def stop(self):
        """Stop the sync manager"""
        self.is_running = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Cancel sync tasks
        for task in self.sync_tasks.values():
            task.cancel()
        
        if self.sync_tasks:
            await asyncio.gather(*self.sync_tasks.values(), return_exceptions=True)
        
        await self.transfer_protocol.stop_server()
        logger.info("Memory Sync Manager stopped")
    
    def add_sync_configuration(self, config: SyncConfiguration) -> str:
        """Add synchronization configuration"""
        session_id = f"sync_{config.target_nova}_{int(datetime.now().timestamp())}"
        
        session = SyncSession(
            session_id=session_id,
            config=config,
            status=SyncStatus.IDLE
        )
        
        self.active_sessions[session_id] = session
        
        # Start sync task if real-time mode
        if config.sync_mode == SyncMode.REAL_TIME:
            self.sync_tasks[session_id] = asyncio.create_task(
                self._real_time_sync_loop(session)
            )
        
        logger.info(f"Added sync configuration for {config.target_nova} (session: {session_id})")
        return session_id
    
    def remove_sync_configuration(self, session_id: str):
        """Remove synchronization configuration"""
        if session_id in self.active_sessions:
            # Cancel sync task
            if session_id in self.sync_tasks:
                self.sync_tasks[session_id].cancel()
                del self.sync_tasks[session_id]
            
            del self.active_sessions[session_id]
            logger.info(f"Removed sync configuration (session: {session_id})")
    
    async def trigger_sync(self, session_id: str, force: bool = False) -> bool:
        """Trigger synchronization for a session"""
        if session_id not in self.active_sessions:
            logger.error(f"Sync session {session_id} not found")
            return False
        
        session = self.active_sessions[session_id]
        
        if session.status == SyncStatus.SYNCING and not force:
            logger.warning(f"Sync session {session_id} already in progress")
            return False
        
        try:
            await self._perform_sync(session)
            return True
        except Exception as e:
            logger.error(f"Sync failed for session {session_id}: {e}")
            session.errors.append(str(e))
            session.status = SyncStatus.ERROR
            return False
    
    async def _perform_sync(self, session: SyncSession):
        """Perform synchronization for a session"""
        session.status = SyncStatus.SYNCING
        session.started_at = datetime.now()
        
        try:
            config = session.config
            
            if config.sync_mode == SyncMode.FULL:
                await self._perform_full_sync(session)
            elif config.sync_mode == SyncMode.INCREMENTAL:
                await self._perform_incremental_sync(session)
            elif config.sync_mode == SyncMode.SELECTIVE:
                await self._perform_selective_sync(session)
            elif config.sync_mode == SyncMode.BACKUP_ONLY:
                await self._perform_backup_sync(session)
            
            session.last_sync = datetime.now()
            session.next_sync = session.last_sync + config.sync_interval
            session.status = SyncStatus.MONITORING if config.sync_mode == SyncMode.REAL_TIME else SyncStatus.IDLE
            
            # Notify callbacks
            await self._notify_sync_complete(session)
            
        except Exception as e:
            session.status = SyncStatus.ERROR
            session.errors.append(str(e))
            logger.error(f"Sync failed: {e}")
            raise
    
    async def _perform_full_sync(self, session: SyncSession):
        """Perform full synchronization"""
        config = session.config
        
        # Get all memories that match privacy and filtering rules
        memories = await self._get_syncable_memories(config)
        
        if not memories:
            logger.info("No memories to sync")
            return
        
        # Create transfer data
        transfer_data = {
            'memories': memories,
            'sync_type': 'full',
            'timestamp': datetime.now().isoformat(),
            'source_nova': self.nova_id
        }
        
        # Perform transfer
        await self._execute_transfer(session, transfer_data, TransferOperation.SYNC_FULL)
        
        # Update statistics
        session.stats['full_sync_count'] = session.stats.get('full_sync_count', 0) + 1
        session.stats['memories_transferred'] = len(memories)
    
    async def _perform_incremental_sync(self, session: SyncSession):
        """Perform incremental synchronization"""
        config = session.config
        
        # Get current snapshot
        current_snapshot = await self._create_memory_snapshot()
        
        # Get previous snapshot
        snapshot_key = f"{self.nova_id}_{config.target_nova}"
        previous_snapshot = self.snapshots.get(snapshot_key)
        
        if previous_snapshot is None:
            # First incremental sync, perform full sync
            logger.info("No previous snapshot found, performing full sync")
            await self._perform_full_sync(session)
            self.snapshots[snapshot_key] = current_snapshot
            return
        
        # Calculate deltas
        deltas = current_snapshot.calculate_deltas(previous_snapshot)
        
        if not deltas:
            logger.info("No changes detected, skipping sync")
            return
        
        # Get full memory data for deltas
        delta_memories = []
        for delta in deltas:
            if delta.operation in ['create', 'update']:
                memory_data = await self._get_memory_by_id(delta.memory_id)
                if memory_data and self.privacy_controller.can_share_memory(
                    memory_data, config.target_nova, self.nova_id
                ):
                    delta_memories.append({
                        'delta': delta.__dict__,
                        'data': memory_data
                    })
            else:  # delete
                delta_memories.append({
                    'delta': delta.__dict__,
                    'data': None
                })
        
        if not delta_memories:
            logger.info("No shareable changes detected, skipping sync")
            return
        
        # Create transfer data
        transfer_data = {
            'deltas': delta_memories,
            'sync_type': 'incremental',
            'timestamp': datetime.now().isoformat(),
            'source_nova': self.nova_id,
            'source_snapshot': current_snapshot.__dict__
        }
        
        # Perform transfer
        await self._execute_transfer(session, transfer_data, TransferOperation.SYNC_INCREMENTAL)
        
        # Update snapshot
        self.snapshots[snapshot_key] = current_snapshot
        
        # Update statistics
        session.stats['incremental_sync_count'] = session.stats.get('incremental_sync_count', 0) + 1
        session.stats['deltas_transferred'] = len(delta_memories)
    
    async def _perform_selective_sync(self, session: SyncSession):
        """Perform selective synchronization"""
        config = session.config
        
        # Get memories matching specific criteria
        memories = await self._get_selective_memories(config)
        
        if not memories:
            logger.info("No memories match selective criteria")
            return
        
        # Create transfer data
        transfer_data = {
            'memories': memories,
            'sync_type': 'selective',
            'selection_criteria': {
                'memory_types': config.memory_types,
                'include_patterns': config.include_patterns,
                'exclude_patterns': config.exclude_patterns,
                'max_age': config.max_memory_age.total_seconds() if config.max_memory_age else None
            },
            'timestamp': datetime.now().isoformat(),
            'source_nova': self.nova_id
        }
        
        # Perform transfer
        await self._execute_transfer(session, transfer_data, TransferOperation.SHARE_SELECTIVE)
        
        # Update statistics
        session.stats['selective_sync_count'] = session.stats.get('selective_sync_count', 0) + 1
        session.stats['memories_transferred'] = len(memories)
    
    async def _perform_backup_sync(self, session: SyncSession):
        """Perform backup synchronization"""
        config = session.config
        
        # Get all memories for backup
        memories = await self._get_all_memories_for_backup()
        
        # Create transfer data
        transfer_data = {
            'memories': memories,
            'sync_type': 'backup',
            'backup_timestamp': datetime.now().isoformat(),
            'source_nova': self.nova_id,
            'full_backup': True
        }
        
        # Perform transfer
        await self._execute_transfer(session, transfer_data, TransferOperation.BACKUP)
        
        # Update statistics
        session.stats['backup_count'] = session.stats.get('backup_count', 0) + 1
        session.stats['memories_backed_up'] = len(memories)
    
    async def _execute_transfer(self, session: SyncSession, transfer_data: Dict[str, Any], 
                              operation: TransferOperation):
        """Execute the actual transfer"""
        config = session.config
        
        # Apply bandwidth optimization
        data_size = len(json.dumps(transfer_data))
        chunk_size = self.bandwidth_optimizer.get_optimal_chunk_size(config.target_nova)
        use_compression = self.bandwidth_optimizer.should_enable_compression(config.target_nova, data_size)
        
        options = {
            'chunk_size': chunk_size,
            'compression_enabled': use_compression and config.compression_enabled,
            'encryption_enabled': config.encryption_enabled,
            'bandwidth_limit': config.bandwidth_limit,
            'conflict_resolution': config.conflict_resolution.value
        }
        
        start_time = datetime.now()
        
        # Execute transfer
        transfer_session = await self.transfer_protocol.initiate_transfer(
            target_nova=config.target_nova,
            target_host=config.target_host,
            target_port=config.target_port,
            operation=operation,
            memory_data=transfer_data,
            options=options
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Record statistics
        self.bandwidth_optimizer.record_transfer_stats(
            config.target_nova,
            transfer_session.bytes_transferred,
            duration,
            transfer_session.compression_ratio
        )
        
        # Update session stats
        session.stats.update({
            'last_transfer_bytes': transfer_session.bytes_transferred,
            'last_transfer_duration': duration,
            'last_compression_ratio': transfer_session.compression_ratio,
            'total_bytes_transferred': session.stats.get('total_bytes_transferred', 0) + transfer_session.bytes_transferred
        })
        
        logger.info(f"Transfer completed: {transfer_session.bytes_transferred} bytes in {duration:.2f}s")
    
    async def _get_syncable_memories(self, config: SyncConfiguration) -> List[Dict[str, Any]]:
        """Get memories that can be synchronized"""
        query = {}
        
        # Apply memory type filter
        if config.memory_types:
            query['memory_types'] = config.memory_types
        
        # Apply age filter
        if config.max_memory_age:
            query['max_age'] = config.max_memory_age.total_seconds()
        
        # Get memories
        response = await self.memory_api.recall(self.nova_id, query, limit=10000)
        
        if not response.success:
            logger.error(f"Failed to retrieve memories: {response.errors}")
            return []
        
        memories = response.data.get('memories', [])
        
        # Apply privacy filtering
        syncable_memories = []
        for memory in memories:
            if self.privacy_controller.can_share_memory(memory, config.target_nova, self.nova_id):
                # Apply include/exclude patterns
                if self._matches_patterns(memory, config.include_patterns, config.exclude_patterns):
                    syncable_memories.append(memory)
        
        return syncable_memories
    
    async def _get_selective_memories(self, config: SyncConfiguration) -> List[Dict[str, Any]]:
        """Get memories for selective synchronization"""
        # Similar to _get_syncable_memories but with more specific criteria
        return await self._get_syncable_memories(config)
    
    async def _get_all_memories_for_backup(self) -> List[Dict[str, Any]]:
        """Get all memories for backup purposes"""
        response = await self.memory_api.recall(self.nova_id, limit=100000)
        
        if not response.success:
            logger.error(f"Failed to retrieve memories for backup: {response.errors}")
            return []
        
        return response.data.get('memories', [])
    
    async def _get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get specific memory by ID"""
        response = await self.memory_api.recall(self.nova_id, {'memory_id': memory_id}, limit=1)
        
        if not response.success or not response.data.get('memories'):
            return None
        
        return response.data['memories'][0]
    
    async def _create_memory_snapshot(self) -> MemorySnapshot:
        """Create snapshot of current memory state"""
        response = await self.memory_api.recall(self.nova_id, limit=100000)
        
        if not response.success:
            logger.error(f"Failed to create memory snapshot: {response.errors}")
            return MemorySnapshot(
                nova_id=self.nova_id,
                timestamp=datetime.now(),
                memory_checksums={},
                total_count=0,
                last_modified={},
                vector_clock=VectorClock({self.nova_id: int(datetime.now().timestamp())})
            )
        
        memories = response.data.get('memories', [])
        checksums = {}
        last_modified = {}
        
        for memory in memories:
            memory_id = memory.get('id', '')
            if memory_id:
                # Create checksum from memory content
                memory_str = json.dumps(memory, sort_keys=True)
                checksums[memory_id] = hashlib.sha256(memory_str.encode()).hexdigest()
                
                # Extract timestamp
                if 'timestamp' in memory:
                    try:
                        last_modified[memory_id] = datetime.fromisoformat(memory['timestamp'])
                    except:
                        last_modified[memory_id] = datetime.now()
                else:
                    last_modified[memory_id] = datetime.now()
        
        return MemorySnapshot(
            nova_id=self.nova_id,
            timestamp=datetime.now(),
            memory_checksums=checksums,
            total_count=len(memories),
            last_modified=last_modified,
            vector_clock=VectorClock({self.nova_id: int(datetime.now().timestamp())})
        )
    
    def _matches_patterns(self, memory: Dict[str, Any], include_patterns: List[str], 
                         exclude_patterns: List[str]) -> bool:
        """Check if memory matches include/exclude patterns"""
        memory_text = str(memory).lower()
        
        # Check exclude patterns first
        for pattern in exclude_patterns:
            if pattern.lower() in memory_text:
                return False
        
        # If no include patterns, include by default
        if not include_patterns:
            return True
        
        # Check include patterns
        for pattern in include_patterns:
            if pattern.lower() in memory_text:
                return True
        
        return False
    
    async def _real_time_sync_loop(self, session: SyncSession):
        """Real-time synchronization loop"""
        logger.info(f"Starting real-time sync loop for {session.config.target_nova}")
        
        while self.is_running and session.session_id in self.active_sessions:
            try:
                await self._perform_sync(session)
                await asyncio.sleep(session.config.sync_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Real-time sync error: {e}")
                session.errors.append(str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retry
        
        logger.info(f"Real-time sync loop ended for {session.config.target_nova}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for session in self.active_sessions.values():
                    if (session.status == SyncStatus.IDLE and
                        session.next_sync and
                        current_time >= session.next_sync):
                        
                        # Trigger scheduled sync
                        asyncio.create_task(self._perform_sync(session))
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _notify_sync_complete(self, session: SyncSession):
        """Notify callbacks of sync completion"""
        for callback_ref in self.sync_callbacks[:]:  # Copy to avoid modification during iteration
            callback = callback_ref()
            if callback is None:
                self.sync_callbacks.remove(callback_ref)
            else:
                try:
                    await callback(session)
                except Exception as e:
                    logger.error(f"Sync callback error: {e}")
    
    def add_sync_callback(self, callback):
        """Add callback for sync events"""
        self.sync_callbacks.append(weakref.WeakMethod(callback))
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get overall sync status"""
        return {
            'nova_id': self.nova_id,
            'is_running': self.is_running,
            'active_sessions': len(self.active_sessions),
            'sessions': [session.to_dict() for session in self.active_sessions.values()]
        }

# Example usage
async def example_memory_sync():
    """Example memory synchronization setup"""
    
    # Initialize memory API
    memory_api = NovaMemoryAPI()
    await memory_api.initialize()
    
    # Create sync manager
    sync_manager = MemorySyncManager('PRIME', memory_api)
    await sync_manager.start()
    
    try:
        # Configure privacy rules
        sync_manager.privacy_controller.add_team_membership('core_team', {'PRIME', 'AXIOM', 'NEXUS'})
        sync_manager.privacy_controller.set_privacy_rule('user_conversation', PrivacyLevel.TEAM)
        sync_manager.privacy_controller.set_privacy_rule('system_internal', PrivacyLevel.PRIVATE)
        
        # Add sync configuration
        config = SyncConfiguration(
            target_nova='AXIOM',
            target_host='axiom.nova.local',
            target_port=8443,
            sync_mode=SyncMode.INCREMENTAL,
            sync_direction=SyncDirection.BIDIRECTIONAL,
            sync_interval=timedelta(minutes=5),
            memory_types=['conversation', 'learning'],
            privacy_levels=[PrivacyLevel.PUBLIC, PrivacyLevel.TEAM]
        )
        
        session_id = sync_manager.add_sync_configuration(config)
        
        # Trigger initial sync
        success = await sync_manager.trigger_sync(session_id)
        print(f"Initial sync success: {success}")
        
        # Monitor for a while
        await asyncio.sleep(30)
        
        # Check status
        status = sync_manager.get_sync_status()
        print(f"Sync status: {json.dumps(status, indent=2)}")
        
    finally:
        await sync_manager.stop()
        await memory_api.shutdown()

if __name__ == "__main__":
    asyncio.run(example_memory_sync())