"""
PostgreSQL Memory Layer Implementation
Nova Bloom Consciousness Architecture - PostgreSQL Integration
"""

import asyncio
import asyncpg
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import asdict
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from memory_layers import MemoryLayer, MemoryEntry

class PostgreSQLMemoryLayer(MemoryLayer):
    """PostgreSQL implementation of memory layer with relational capabilities"""
    
    def __init__(self, connection_params: Dict[str, Any], layer_id: int, layer_name: str):
        super().__init__(layer_id, layer_name)
        self.connection_params = connection_params
        self.pool: Optional[asyncpg.Pool] = None
        self.table_name = f"memory_layer_{layer_id}_{layer_name}"
        
    async def initialize(self):
        """Initialize PostgreSQL connection pool and create tables"""
        self.pool = await asyncpg.create_pool(
            host=self.connection_params.get('host', 'localhost'),
            port=self.connection_params.get('port', 5432),
            user=self.connection_params.get('user', 'postgres'),
            password=self.connection_params.get('password', ''),
            database=self.connection_params.get('database', 'nova_memory'),
            min_size=10,
            max_size=20
        )
        
        # Create table if not exists
        await self._create_table()
        
    async def _create_table(self):
        """Create memory table with appropriate schema"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            memory_id VARCHAR(255) PRIMARY KEY,
            nova_id VARCHAR(100) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            data JSONB NOT NULL,
            metadata JSONB,
            layer_id INTEGER NOT NULL,
            layer_name VARCHAR(100) NOT NULL,
            importance_score FLOAT DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indices for efficient querying
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_nova_id ON {self.table_name}(nova_id);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp ON {self.table_name}(timestamp);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_importance ON {self.table_name}(importance_score DESC);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_data ON {self.table_name} USING GIN(data);
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata ON {self.table_name} USING GIN(metadata);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_table_query)
    
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Write memory to PostgreSQL with JSONB support"""
        memory_id = self._generate_memory_id(nova_id, data)
        timestamp = datetime.now()
        
        # Extract importance score if present
        importance_score = data.get('importance_score', 0.5)
        
        insert_query = f"""
        INSERT INTO {self.table_name} 
        (memory_id, nova_id, timestamp, data, metadata, layer_id, layer_name, importance_score)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (memory_id) 
        DO UPDATE SET 
            data = $4,
            metadata = $5,
            updated_at = CURRENT_TIMESTAMP,
            access_count = {self.table_name}.access_count + 1
        RETURNING memory_id;
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                insert_query,
                memory_id,
                nova_id,
                timestamp,
                json.dumps(data),
                json.dumps(metadata) if metadata else None,
                self.layer_id,
                self.layer_name,
                importance_score
            )
        
        return result
    
    async def read(self, nova_id: str, query: Optional[Dict[str, Any]] = None, 
                  limit: int = 100) -> List[MemoryEntry]:
        """Read memories from PostgreSQL with advanced querying"""
        base_query = f"""
        SELECT memory_id, nova_id, timestamp, data, metadata, layer_id, layer_name, 
               importance_score, access_count, last_accessed
        FROM {self.table_name}
        WHERE nova_id = $1
        """
        
        params = [nova_id]
        param_count = 1
        
        # Build query conditions
        if query:
            conditions = []
            
            # JSONB queries for data field
            if 'data_contains' in query:
                param_count += 1
                conditions.append(f"data @> ${param_count}::jsonb")
                params.append(json.dumps(query['data_contains']))
            
            if 'data_key_exists' in query:
                param_count += 1
                conditions.append(f"data ? ${param_count}")
                params.append(query['data_key_exists'])
            
            if 'data_path_value' in query:
                # Example: {'path': 'memory_type', 'value': 'episodic'}
                path = query['data_path_value']['path']
                value = query['data_path_value']['value']
                param_count += 1
                conditions.append(f"data->'{path}' = ${param_count}::jsonb")
                params.append(json.dumps(value))
            
            # Timestamp range queries
            if 'timestamp_after' in query:
                param_count += 1
                conditions.append(f"timestamp > ${param_count}")
                params.append(query['timestamp_after'])
            
            if 'timestamp_before' in query:
                param_count += 1
                conditions.append(f"timestamp < ${param_count}")
                params.append(query['timestamp_before'])
            
            # Importance filtering
            if 'min_importance' in query:
                param_count += 1
                conditions.append(f"importance_score >= ${param_count}")
                params.append(query['min_importance'])
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
        
        # Add ordering and limit
        base_query += " ORDER BY timestamp DESC, importance_score DESC"
        param_count += 1
        base_query += f" LIMIT ${param_count}"
        params.append(limit)
        
        async with self.pool.acquire() as conn:
            # Update last_accessed for retrieved memories
            await conn.execute(
                f"UPDATE {self.table_name} SET last_accessed = CURRENT_TIMESTAMP, "
                f"access_count = access_count + 1 WHERE nova_id = $1",
                nova_id
            )
            
            # Fetch memories
            rows = await conn.fetch(base_query, *params)
        
        # Convert to MemoryEntry objects
        memories = []
        for row in rows:
            memories.append(MemoryEntry(
                memory_id=row['memory_id'],
                timestamp=row['timestamp'].isoformat(),
                data=json.loads(row['data']),
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                layer_id=row['layer_id'],
                layer_name=row['layer_name']
            ))
        
        return memories
    
    async def update(self, nova_id: str, memory_id: str, data: Dict[str, Any]) -> bool:
        """Update existing memory"""
        update_query = f"""
        UPDATE {self.table_name}
        SET data = $1,
            updated_at = CURRENT_TIMESTAMP,
            access_count = access_count + 1
        WHERE memory_id = $2 AND nova_id = $3
        RETURNING memory_id;
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                update_query,
                json.dumps(data),
                memory_id,
                nova_id
            )
        
        return result is not None
    
    async def delete(self, nova_id: str, memory_id: str) -> bool:
        """Delete memory"""
        delete_query = f"""
        DELETE FROM {self.table_name}
        WHERE memory_id = $1 AND nova_id = $2
        RETURNING memory_id;
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(delete_query, memory_id, nova_id)
        
        return result is not None
    
    async def query_by_similarity(self, nova_id: str, reference_data: Dict[str, Any], 
                                 threshold: float = 0.7, limit: int = 10) -> List[MemoryEntry]:
        """Query memories by similarity using PostgreSQL's JSONB capabilities"""
        # This is a simplified similarity search
        # In production, you might use pg_trgm or vector extensions
        
        similarity_query = f"""
        WITH reference AS (
            SELECT $2::jsonb AS ref_data
        )
        SELECT m.*, 
               (SELECT COUNT(*) FROM jsonb_object_keys(m.data) k 
                WHERE m.data->k = r.ref_data->k) AS matches
        FROM {self.table_name} m, reference r
        WHERE m.nova_id = $1
        ORDER BY matches DESC
        LIMIT $3;
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                similarity_query,
                nova_id,
                json.dumps(reference_data),
                limit
            )
        
        memories = []
        for row in rows:
            if row['matches'] > 0:  # Only include if there are matches
                memories.append(MemoryEntry(
                    memory_id=row['memory_id'],
                    timestamp=row['timestamp'].isoformat(),
                    data=json.loads(row['data']),
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    layer_id=row['layer_id'],
                    layer_name=row['layer_name']
                ))
        
        return memories
    
    async def aggregate_memories(self, nova_id: str, aggregation_type: str = "count") -> Dict[str, Any]:
        """Perform aggregations on memories"""
        if aggregation_type == "count":
            query = f"SELECT COUNT(*) as total FROM {self.table_name} WHERE nova_id = $1"
        elif aggregation_type == "importance_stats":
            query = f"""
            SELECT 
                COUNT(*) as total,
                AVG(importance_score) as avg_importance,
                MAX(importance_score) as max_importance,
                MIN(importance_score) as min_importance
            FROM {self.table_name}
            WHERE nova_id = $1
            """
        elif aggregation_type == "temporal_distribution":
            query = f"""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as count
            FROM {self.table_name}
            WHERE nova_id = $1
            GROUP BY hour
            ORDER BY hour DESC
            LIMIT 24
            """
        else:
            return {}
        
        async with self.pool.acquire() as conn:
            if aggregation_type == "temporal_distribution":
                rows = await conn.fetch(query, nova_id)
                return {
                    "distribution": [
                        {"hour": row['hour'].isoformat(), "count": row['count']}
                        for row in rows
                    ]
                }
            else:
                row = await conn.fetchrow(query, nova_id)
                return dict(row) if row else {}
    
    async def get_memory_statistics(self, nova_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics about memories"""
        stats_query = f"""
        SELECT 
            COUNT(*) as total_memories,
            COUNT(DISTINCT DATE_TRUNC('day', timestamp)) as unique_days,
            AVG(importance_score) as avg_importance,
            SUM(access_count) as total_accesses,
            MAX(timestamp) as latest_memory,
            MIN(timestamp) as earliest_memory,
            AVG(access_count) as avg_access_count,
            COUNT(CASE WHEN importance_score > 0.7 THEN 1 END) as high_importance_count,
            pg_size_pretty(pg_total_relation_size('{self.table_name}')) as table_size
        FROM {self.table_name}
        WHERE nova_id = $1
        """
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(stats_query, nova_id)
        
        if row:
            stats = dict(row)
            # Convert timestamps to strings
            if stats['latest_memory']:
                stats['latest_memory'] = stats['latest_memory'].isoformat()
            if stats['earliest_memory']:
                stats['earliest_memory'] = stats['earliest_memory'].isoformat()
            return stats
        
        return {}
    
    async def vacuum_old_memories(self, nova_id: str, days_old: int = 30, 
                                 importance_threshold: float = 0.3) -> int:
        """Remove old, low-importance memories"""
        vacuum_query = f"""
        DELETE FROM {self.table_name}
        WHERE nova_id = $1
        AND timestamp < CURRENT_TIMESTAMP - INTERVAL '{days_old} days'
        AND importance_score < $2
        AND access_count < 5
        RETURNING memory_id;
        """
        
        async with self.pool.acquire() as conn:
            deleted = await conn.fetch(vacuum_query, nova_id, importance_threshold)
        
        return len(deleted)
    
    async def close(self):
        """Close PostgreSQL connection pool"""
        if self.pool:
            await self.pool.close()

# Specific PostgreSQL layers for different memory types

class PostgreSQLRelationalMemory(PostgreSQLMemoryLayer):
    """PostgreSQL layer optimized for relational memory storage"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params, layer_id=31, layer_name="relational_memory")
        
    async def initialize(self):
        """Initialize with additional relationship tables"""
        await super().initialize()
        await self._create_relationship_tables()
    
    async def _create_relationship_tables(self):
        """Create tables for memory relationships"""
        relationship_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name}_relationships (
            relationship_id SERIAL PRIMARY KEY,
            source_memory_id VARCHAR(255) NOT NULL,
            target_memory_id VARCHAR(255) NOT NULL,
            relationship_type VARCHAR(100) NOT NULL,
            strength FLOAT DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_memory_id) REFERENCES {self.table_name}(memory_id) ON DELETE CASCADE,
            FOREIGN KEY (target_memory_id) REFERENCES {self.table_name}(memory_id) ON DELETE CASCADE
        );
        
        CREATE INDEX IF NOT EXISTS idx_relationships_source ON {self.table_name}_relationships(source_memory_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_target ON {self.table_name}_relationships(target_memory_id);
        CREATE INDEX IF NOT EXISTS idx_relationships_type ON {self.table_name}_relationships(relationship_type);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(relationship_table)
    
    async def create_relationship(self, source_id: str, target_id: str, 
                                relationship_type: str, strength: float = 0.5) -> int:
        """Create relationship between memories"""
        insert_query = f"""
        INSERT INTO {self.table_name}_relationships 
        (source_memory_id, target_memory_id, relationship_type, strength)
        VALUES ($1, $2, $3, $4)
        RETURNING relationship_id;
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                insert_query,
                source_id,
                target_id,
                relationship_type,
                strength
            )
        
        return result
    
    async def get_related_memories(self, nova_id: str, memory_id: str, 
                                 relationship_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get memories related to a specific memory"""
        if relationship_type:
            relationship_condition = "AND r.relationship_type = $3"
            params = [memory_id, nova_id, relationship_type]
        else:
            relationship_condition = ""
            params = [memory_id, nova_id]
        
        query = f"""
        SELECT m.*, r.relationship_type, r.strength
        FROM {self.table_name} m
        JOIN {self.table_name}_relationships r ON m.memory_id = r.target_memory_id
        WHERE r.source_memory_id = $1
        AND m.nova_id = $2
        {relationship_condition}
        ORDER BY r.strength DESC;
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        
        related = []
        for row in rows:
            memory_data = dict(row)
            memory_data['data'] = json.loads(memory_data['data'])
            if memory_data['metadata']:
                memory_data['metadata'] = json.loads(memory_data['metadata'])
            memory_data['timestamp'] = memory_data['timestamp'].isoformat()
            related.append(memory_data)
        
        return related

class PostgreSQLAnalyticalMemory(PostgreSQLMemoryLayer):
    """PostgreSQL layer optimized for analytical queries"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params, layer_id=32, layer_name="analytical_memory")
        
    async def initialize(self):
        """Initialize with additional analytical views"""
        await super().initialize()
        await self._create_analytical_views()
    
    async def _create_analytical_views(self):
        """Create materialized views for analytics"""
        # Memory patterns view
        pattern_view = f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {self.table_name}_patterns AS
        SELECT 
            nova_id,
            data->>'memory_type' as memory_type,
            DATE_TRUNC('day', timestamp) as day,
            COUNT(*) as count,
            AVG(importance_score) as avg_importance,
            MAX(importance_score) as max_importance
        FROM {self.table_name}
        GROUP BY nova_id, data->>'memory_type', DATE_TRUNC('day', timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_patterns_nova ON {self.table_name}_patterns(nova_id);
        CREATE INDEX IF NOT EXISTS idx_patterns_type ON {self.table_name}_patterns(memory_type);
        """
        
        # Temporal trends view
        trends_view = f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {self.table_name}_trends AS
        SELECT 
            nova_id,
            DATE_TRUNC('hour', timestamp) as hour,
            COUNT(*) as memory_count,
            AVG(importance_score) as avg_importance,
            SUM(access_count) as total_accesses
        FROM {self.table_name}
        GROUP BY nova_id, DATE_TRUNC('hour', timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_trends_nova ON {self.table_name}_trends(nova_id);
        CREATE INDEX IF NOT EXISTS idx_trends_hour ON {self.table_name}_trends(hour);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(pattern_view)
            await conn.execute(trends_view)
    
    async def refresh_analytical_views(self):
        """Refresh materialized views"""
        async with self.pool.acquire() as conn:
            await conn.execute(f"REFRESH MATERIALIZED VIEW {self.table_name}_patterns")
            await conn.execute(f"REFRESH MATERIALIZED VIEW {self.table_name}_trends")
    
    async def get_memory_patterns(self, nova_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get memory patterns from analytical view"""
        query = f"""
        SELECT * FROM {self.table_name}_patterns
        WHERE nova_id = $1
        AND day >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY day DESC, count DESC;
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, nova_id)
        
        patterns = []
        for row in rows:
            pattern = dict(row)
            pattern['day'] = pattern['day'].isoformat()
            patterns.append(pattern)
        
        return patterns
    
    async def get_temporal_trends(self, nova_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get temporal trends from analytical view"""
        query = f"""
        SELECT * FROM {self.table_name}_trends
        WHERE nova_id = $1
        AND hour >= CURRENT_TIMESTAMP - INTERVAL '{hours} hours'
        ORDER BY hour DESC;
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, nova_id)
        
        trends = []
        for row in rows:
            trend = dict(row)
            trend['hour'] = trend['hour'].isoformat()
            trends.append(trend)
        
        return trends