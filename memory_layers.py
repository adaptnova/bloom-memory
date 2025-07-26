#!/usr/bin/env python3
"""
Nova Memory System - Base Memory Layer Classes
Implements database-specific memory layer abstractions
"""

import json
import uuid
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryScope(Enum):
    """Memory scope definitions"""
    VOLATILE = "volatile"        # Lost on session end
    SESSION = "session"          # Persists for session
    TEMPORARY = "temporary"      # Short-term storage
    PERSISTENT = "persistent"    # Long-term storage
    PERMANENT = "permanent"      # Never deleted

class MemoryImportance(Enum):
    """Memory importance levels"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    MINIMAL = 0.1

@dataclass
class MemoryEntry:
    """Standard memory entry structure"""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nova_id: str = ""
    layer_id: int = 0
    layer_name: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    context: str = "general"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'memory_id': self.memory_id,
            'nova_id': self.nova_id,
            'layer_id': self.layer_id,
            'layer_name': self.layer_name,
            'timestamp': self.timestamp,
            'data': self.data,
            'metadata': self.metadata,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'context': self.context,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        return cls(**data)

class MemoryLayer(ABC):
    """
    Abstract base class for all memory layers
    Defines the interface that all memory layers must implement
    """
    
    def __init__(self, layer_id: int, layer_name: str, database: str, 
                 capacity: Optional[int] = None, retention: Optional[timedelta] = None,
                 scope: MemoryScope = MemoryScope.PERSISTENT):
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.database = database
        self.capacity = capacity
        self.retention = retention
        self.scope = scope
        self.stats = {
            'total_writes': 0,
            'total_reads': 0,
            'total_updates': 0,
            'total_deletes': 0,
            'last_operation': None
        }
        
    @abstractmethod
    async def initialize(self, connection):
        """Initialize the memory layer with database connection"""
        pass
        
    @abstractmethod
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   importance: float = 0.5, context: str = "general",
                   tags: List[str] = None) -> str:
        """Write memory to layer"""
        pass
        
    @abstractmethod
    async def read(self, nova_id: str, query: Optional[Dict[str, Any]] = None,
                  limit: int = 100, offset: int = 0) -> List[MemoryEntry]:
        """Read memories from layer"""
        pass
        
    @abstractmethod
    async def update(self, nova_id: str, memory_id: str, 
                    data: Dict[str, Any]) -> bool:
        """Update existing memory"""
        pass
        
    @abstractmethod
    async def delete(self, nova_id: str, memory_id: str) -> bool:
        """Delete memory (if allowed by retention policy)"""
        pass
        
    async def search(self, nova_id: str, search_query: str, 
                    limit: int = 50) -> List[MemoryEntry]:
        """Search memories (optional implementation)"""
        return []
        
    async def get_by_id(self, nova_id: str, memory_id: str) -> Optional[MemoryEntry]:
        """Get specific memory by ID"""
        results = await self.read(nova_id, {'memory_id': memory_id}, limit=1)
        return results[0] if results else None
        
    async def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        return {
            'layer_id': self.layer_id,
            'layer_name': self.layer_name,
            'database': self.database,
            'stats': self.stats,
            'capacity': self.capacity,
            'scope': self.scope.value
        }
        
    async def cleanup(self):
        """Cleanup old memories based on retention policy"""
        if self.retention and self.scope != MemoryScope.PERMANENT:
            cutoff_time = datetime.now() - self.retention
            # Implementation depends on specific database
            pass
            
    def _update_stats(self, operation: str):
        """Update operation statistics"""
        self.stats[f'total_{operation}s'] += 1
        self.stats['last_operation'] = {
            'type': operation,
            'timestamp': datetime.now().isoformat()
        }

class DragonflyMemoryLayer(MemoryLayer):
    """
    DragonflyDB implementation for real-time memory layers
    Used for layers 1-10 (immediate and short-term storage)
    """
    
    def __init__(self, layer_id: int, layer_name: str, **kwargs):
        super().__init__(layer_id, layer_name, "dragonfly", **kwargs)
        self.connection = None
        self.stream_key_template = "nova:{nova_id}:{layer_name}"
        
    async def initialize(self, connection):
        """Initialize with DragonflyDB connection"""
        self.connection = connection
        logger.info(f"Initialized DragonflyDB layer: {self.layer_name}")
        
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   importance: float = 0.5, context: str = "general",
                   tags: List[str] = None) -> str:
        """Write to DragonflyDB stream"""
        if not self.connection:
            raise RuntimeError("Layer not initialized")
            
        # Create memory entry
        entry = MemoryEntry(
            nova_id=nova_id,
            layer_id=self.layer_id,
            layer_name=self.layer_name,
            data=data,
            importance=importance,
            context=context,
            tags=tags or []
        )
        
        # Get stream key
        stream_key = self.stream_key_template.format(
            nova_id=nova_id, 
            layer_name=self.layer_name
        )
        
        # Convert entry to stream format
        stream_data = {
            'memory_id': entry.memory_id,
            'timestamp': entry.timestamp,
            'data': json.dumps(entry.data),
            'importance': str(entry.importance),
            'context': entry.context,
            'tags': json.dumps(entry.tags)
        }
        
        # Add to stream
        message_id = self.connection.xadd(stream_key, stream_data)
        
        # Update stats
        self._update_stats('write')
        
        # Store full entry in hash for fast lookup
        hash_key = f"{stream_key}:lookup"
        self.connection.hset(hash_key, entry.memory_id, json.dumps(entry.to_dict()))
        
        return entry.memory_id
        
    async def read(self, nova_id: str, query: Optional[Dict[str, Any]] = None,
                  limit: int = 100, offset: int = 0) -> List[MemoryEntry]:
        """Read from DragonflyDB stream"""
        if not self.connection:
            raise RuntimeError("Layer not initialized")
            
        stream_key = self.stream_key_template.format(
            nova_id=nova_id,
            layer_name=self.layer_name
        )
        
        # Read from stream
        if query and 'memory_id' in query:
            # Direct lookup
            hash_key = f"{stream_key}:lookup"
            data = self.connection.hget(hash_key, query['memory_id'])
            if data:
                return [MemoryEntry.from_dict(json.loads(data))]
            return []
            
        # Stream range query
        messages = self.connection.xrevrange(stream_key, count=limit)
        
        entries = []
        for message_id, data in messages:
            entry_data = {
                'memory_id': data.get('memory_id'),
                'nova_id': nova_id,
                'layer_id': self.layer_id,
                'layer_name': self.layer_name,
                'timestamp': data.get('timestamp'),
                'data': json.loads(data.get('data', '{}')),
                'importance': float(data.get('importance', 0.5)),
                'context': data.get('context', 'general'),
                'tags': json.loads(data.get('tags', '[]'))
            }
            entries.append(MemoryEntry.from_dict(entry_data))
            
        # Update stats
        self._update_stats('read')
        
        return entries[offset:offset+limit] if offset else entries
        
    async def update(self, nova_id: str, memory_id: str, 
                    data: Dict[str, Any]) -> bool:
        """Update memory in hash lookup"""
        if not self.connection:
            raise RuntimeError("Layer not initialized")
            
        stream_key = self.stream_key_template.format(
            nova_id=nova_id,
            layer_name=self.layer_name
        )
        hash_key = f"{stream_key}:lookup"
        
        # Get existing entry
        existing = self.connection.hget(hash_key, memory_id)
        if not existing:
            return False
            
        entry = MemoryEntry.from_dict(json.loads(existing))
        entry.data.update(data)
        entry.metadata['updated_at'] = datetime.now().isoformat()
        entry.access_count += 1
        entry.last_accessed = datetime.now().isoformat()
        
        # Update in hash
        self.connection.hset(hash_key, memory_id, json.dumps(entry.to_dict()))
        
        # Update stats
        self._update_stats('update')
        
        return True
        
    async def delete(self, nova_id: str, memory_id: str) -> bool:
        """Delete from hash lookup (stream entries remain for history)"""
        if not self.connection:
            raise RuntimeError("Layer not initialized")
            
        if self.scope == MemoryScope.PERMANENT:
            logger.warning(f"Cannot delete from permanent layer: {self.layer_name}")
            return False
            
        stream_key = self.stream_key_template.format(
            nova_id=nova_id,
            layer_name=self.layer_name
        )
        hash_key = f"{stream_key}:lookup"
        
        result = self.connection.hdel(hash_key, memory_id)
        
        # Update stats
        self._update_stats('delete')
        
        return bool(result)

class ClickHouseMemoryLayer(MemoryLayer):
    """
    ClickHouse implementation for time-series memory layers
    Used for analytics and temporal patterns
    """
    
    def __init__(self, layer_id: int, layer_name: str, **kwargs):
        super().__init__(layer_id, layer_name, "clickhouse", **kwargs)
        self.client = None
        self.table_name = f"nova_memory.{layer_name}"
        
    async def initialize(self, connection):
        """Initialize with ClickHouse client"""
        self.client = connection
        
        # Ensure table exists
        self.client.command(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                nova_id String,
                memory_id UUID,
                timestamp DateTime64(3),
                layer_id UInt8,
                layer_name String,
                data String,
                importance Float32,
                context String,
                tags Array(String),
                access_count UInt32 DEFAULT 0,
                last_accessed Nullable(DateTime64(3))
            ) ENGINE = MergeTree()
            ORDER BY (nova_id, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 1 YEAR
        """)
        
        logger.info(f"Initialized ClickHouse layer: {self.layer_name}")
        
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   importance: float = 0.5, context: str = "general",
                   tags: List[str] = None) -> str:
        """Write to ClickHouse table"""
        if not self.client:
            raise RuntimeError("Layer not initialized")
            
        entry = MemoryEntry(
            nova_id=nova_id,
            layer_id=self.layer_id,
            layer_name=self.layer_name,
            data=data,
            importance=importance,
            context=context,
            tags=tags or []
        )
        
        # Insert into ClickHouse
        self.client.insert(
            self.table_name,
            [[
                entry.nova_id,
                entry.memory_id,
                datetime.fromisoformat(entry.timestamp),
                entry.layer_id,
                entry.layer_name,
                json.dumps(entry.data),
                entry.importance,
                entry.context,
                entry.tags,
                0,  # access_count
                None  # last_accessed
            ]],
            column_names=[
                'nova_id', 'memory_id', 'timestamp', 'layer_id', 
                'layer_name', 'data', 'importance', 'context', 
                'tags', 'access_count', 'last_accessed'
            ]
        )
        
        self._update_stats('write')
        return entry.memory_id
        
    async def read(self, nova_id: str, query: Optional[Dict[str, Any]] = None,
                  limit: int = 100, offset: int = 0) -> List[MemoryEntry]:
        """Read from ClickHouse"""
        if not self.client:
            raise RuntimeError("Layer not initialized")
            
        # Build query
        where_clauses = [f"nova_id = '{nova_id}'"]
        
        if query:
            if 'memory_id' in query:
                where_clauses.append(f"memory_id = '{query['memory_id']}'")
            if 'context' in query:
                where_clauses.append(f"context = '{query['context']}'")
            if 'importance_gte' in query:
                where_clauses.append(f"importance >= {query['importance_gte']}")
            if 'timeframe' in query:
                if query['timeframe'] == 'last_hour':
                    where_clauses.append("timestamp > now() - INTERVAL 1 HOUR")
                elif query['timeframe'] == 'last_day':
                    where_clauses.append("timestamp > now() - INTERVAL 1 DAY")
                    
        where_clause = " AND ".join(where_clauses)
        
        sql = f"""
            SELECT 
                nova_id, memory_id, timestamp, layer_id, layer_name,
                data, importance, context, tags, access_count, last_accessed
            FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {limit} OFFSET {offset}
        """
        
        result = self.client.query(sql)
        
        entries = []
        for row in result.result_rows:
            entry_data = {
                'nova_id': row[0],
                'memory_id': str(row[1]),
                'timestamp': row[2].isoformat(),
                'layer_id': row[3],
                'layer_name': row[4],
                'data': json.loads(row[5]),
                'importance': row[6],
                'context': row[7],
                'tags': row[8],
                'access_count': row[9],
                'last_accessed': row[10].isoformat() if row[10] else None
            }
            entries.append(MemoryEntry.from_dict(entry_data))
            
        self._update_stats('read')
        return entries

    async def update(self, nova_id: str, memory_id: str, 
                    data: Dict[str, Any]) -> bool:
        """Update not directly supported in ClickHouse - would need to reinsert"""
        logger.warning("Direct updates not supported in ClickHouse layer")
        return False
        
    async def delete(self, nova_id: str, memory_id: str) -> bool:
        """Delete from ClickHouse (using ALTER TABLE DELETE)"""
        if not self.client:
            raise RuntimeError("Layer not initialized")
            
        if self.scope == MemoryScope.PERMANENT:
            return False
            
        self.client.command(f"""
            ALTER TABLE {self.table_name}
            DELETE WHERE nova_id = '{nova_id}' AND memory_id = '{memory_id}'
        """)
        
        self._update_stats('delete')
        return True

class ArangoMemoryLayer(MemoryLayer):
    """
    ArangoDB implementation for graph-based memory layers
    Used for relationships and connections
    """
    
    def __init__(self, layer_id: int, layer_name: str, **kwargs):
        super().__init__(layer_id, layer_name, "arangodb", **kwargs)
        self.db = None
        self.collection_name = f"memory_{layer_name}"
        
    async def initialize(self, connection):
        """Initialize with ArangoDB database"""
        self.db = connection
        
        # Create collection if not exists
        if not self.db.has_collection(self.collection_name):
            self.db.create_collection(self.collection_name)
            
        # Create indexes
        collection = self.db.collection(self.collection_name)
        collection.add_hash_index(fields=['nova_id', 'memory_id'])
        collection.add_skiplist_index(fields=['nova_id', 'timestamp'])
        
        logger.info(f"Initialized ArangoDB layer: {self.layer_name}")
        
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   importance: float = 0.5, context: str = "general",
                   tags: List[str] = None) -> str:
        """Write to ArangoDB collection"""
        if not self.db:
            raise RuntimeError("Layer not initialized")
            
        entry = MemoryEntry(
            nova_id=nova_id,
            layer_id=self.layer_id,
            layer_name=self.layer_name,
            data=data,
            importance=importance,
            context=context,
            tags=tags or []
        )
        
        collection = self.db.collection(self.collection_name)
        doc = entry.to_dict()
        doc['_key'] = entry.memory_id
        
        collection.insert(doc)
        
        self._update_stats('write')
        return entry.memory_id
        
    async def read(self, nova_id: str, query: Optional[Dict[str, Any]] = None,
                  limit: int = 100, offset: int = 0) -> List[MemoryEntry]:
        """Read from ArangoDB"""
        if not self.db:
            raise RuntimeError("Layer not initialized")
            
        # Build AQL query
        aql_query = f"""
            FOR doc IN {self.collection_name}
            FILTER doc.nova_id == @nova_id
        """
        
        bind_vars = {'nova_id': nova_id}
        
        if query:
            if 'memory_id' in query:
                aql_query += " FILTER doc.memory_id == @memory_id"
                bind_vars['memory_id'] = query['memory_id']
            if 'context' in query:
                aql_query += " FILTER doc.context == @context"
                bind_vars['context'] = query['context']
                
        aql_query += f"""
            SORT doc.timestamp DESC
            LIMIT {offset}, {limit}
            RETURN doc
        """
        
        cursor = self.db.aql.execute(aql_query, bind_vars=bind_vars)
        
        entries = []
        for doc in cursor:
            # Remove ArangoDB internal fields
            doc.pop('_id', None)
            doc.pop('_key', None)
            doc.pop('_rev', None)
            entries.append(MemoryEntry.from_dict(doc))
            
        self._update_stats('read')
        return entries

    async def update(self, nova_id: str, memory_id: str, 
                    data: Dict[str, Any]) -> bool:
        """Update document in ArangoDB"""
        if not self.db:
            raise RuntimeError("Layer not initialized")
            
        collection = self.db.collection(self.collection_name)
        
        try:
            doc = collection.get(memory_id)
            doc['data'].update(data)
            doc['access_count'] = doc.get('access_count', 0) + 1
            doc['last_accessed'] = datetime.now().isoformat()
            
            collection.update(doc)
            self._update_stats('update')
            return True
        except:
            return False
            
    async def delete(self, nova_id: str, memory_id: str) -> bool:
        """Delete from ArangoDB"""
        if not self.db:
            raise RuntimeError("Layer not initialized")
            
        if self.scope == MemoryScope.PERMANENT:
            return False
            
        collection = self.db.collection(self.collection_name)
        
        try:
            collection.delete(memory_id)
            self._update_stats('delete')
            return True
        except:
            return False

# Additional database implementations would follow similar patterns...
# PostgreSQLMemoryLayer, CouchDBMemoryLayer, MeiliSearchMemoryLayer, etc.

class MemoryLayerFactory:
    """Factory for creating appropriate memory layer instances"""
    
    DATABASE_LAYER_MAP = {
        'dragonfly': DragonflyMemoryLayer,
        'clickhouse': ClickHouseMemoryLayer,
        'arangodb': ArangoMemoryLayer,
        # Add more as implemented
    }
    
    @classmethod
    def create_layer(cls, layer_id: int, layer_name: str, database: str, 
                    **kwargs) -> MemoryLayer:
        """Create a memory layer instance for the specified database"""
        layer_class = cls.DATABASE_LAYER_MAP.get(database)
        
        if not layer_class:
            raise ValueError(f"Unsupported database: {database}")
            
        return layer_class(layer_id, layer_name, **kwargs)

# Example usage
async def test_memory_layers():
    """Test memory layer implementations"""
    
    # Create layers
    working_memory = MemoryLayerFactory.create_layer(
        3, "working_memory", "dragonfly",
        capacity=100, 
        retention=timedelta(minutes=10),
        scope=MemoryScope.SESSION
    )
    
    temporal_patterns = MemoryLayerFactory.create_layer(
        26, "temporal_patterns", "clickhouse",
        scope=MemoryScope.PERSISTENT
    )
    
    memory_relationships = MemoryLayerFactory.create_layer(
        41, "memory_relationships", "arangodb",
        scope=MemoryScope.PERMANENT
    )
    
    # Initialize with connections (would come from database pool)
    # await working_memory.initialize(dragonfly_connection)
    # await temporal_patterns.initialize(clickhouse_client)
    # await memory_relationships.initialize(arangodb_database)
    
    # Test operations
    # memory_id = await working_memory.write("bloom", {"thought": "Testing memory system"})
    # memories = await working_memory.read("bloom", limit=10)
    
    logger.info("Memory layer tests completed")

if __name__ == "__main__":
    asyncio.run(test_memory_layers())