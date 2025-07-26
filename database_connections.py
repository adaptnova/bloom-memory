#!/usr/bin/env python3
"""
Nova Memory System - Multi-Database Connection Manager
Implements connection pooling for all operational databases
Based on /data/.claude/CURRENT_DATABASE_CONNECTIONS.md
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Database clients
import redis
import asyncio_redis
import clickhouse_connect
from arango import ArangoClient
import couchdb
import asyncpg
import psycopg2
from psycopg2 import pool
import meilisearch
import pymongo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    name: str
    host: str
    port: int
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 10
    max_pool_size: int = 100
    
class NovaDatabasePool:
    """
    Multi-database connection pool manager for Nova Memory System
    Manages connections to all operational databases
    """
    
    def __init__(self):
        self.connections = {}
        self.pools = {}
        self.health_status = {}
        self.configs = self._load_database_configs()
        
    def _load_database_configs(self) -> Dict[str, DatabaseConfig]:
        """Load database configurations based on operational status"""
        return {
            'dragonfly': DatabaseConfig(
                name='dragonfly',
                host='localhost',
                port=16381,  # APEX port
                pool_size=20,
                max_pool_size=200
            ),
            'clickhouse': DatabaseConfig(
                name='clickhouse',
                host='localhost',
                port=18123,  # APEX port
                pool_size=15,
                max_pool_size=150
            ),
            'arangodb': DatabaseConfig(
                name='arangodb',
                host='localhost',
                port=19600,  # APEX port
                pool_size=10,
                max_pool_size=100
            ),
            'couchdb': DatabaseConfig(
                name='couchdb',
                host='localhost',
                port=5984,  # Standard port maintained by APEX
                pool_size=10,
                max_pool_size=100
            ),
            'postgresql': DatabaseConfig(
                name='postgresql',
                host='localhost',
                port=15432,  # APEX port
                database='nova_memory',
                username='postgres',
                password='postgres',
                pool_size=15,
                max_pool_size=150
            ),
            'meilisearch': DatabaseConfig(
                name='meilisearch',
                host='localhost',
                port=19640,  # APEX port
                pool_size=5,
                max_pool_size=50
            ),
            'mongodb': DatabaseConfig(
                name='mongodb',
                host='localhost',
                port=17017,  # APEX port
                username='admin',
                password='mongodb',
                pool_size=10,
                max_pool_size=100
            ),
            'redis': DatabaseConfig(
                name='redis',
                host='localhost',
                port=16379,  # APEX port
                pool_size=10,
                max_pool_size=100
            )
        }
        
    async def initialize_all_connections(self):
        """Initialize connections to all databases"""
        logger.info("Initializing Nova database connections...")
        
        # Initialize each database connection
        await self._init_dragonfly()
        await self._init_clickhouse()
        await self._init_arangodb()
        await self._init_couchdb()
        await self._init_postgresql()
        await self._init_meilisearch()
        await self._init_mongodb()
        await self._init_redis()
        
        # Run health checks
        await self.check_all_health()
        
        logger.info(f"Database initialization complete. Status: {self.health_status}")
        
    async def _init_dragonfly(self):
        """Initialize DragonflyDB connection pool"""
        try:
            config = self.configs['dragonfly']
            
            # Synchronous client for immediate operations
            self.connections['dragonfly'] = redis.Redis(
                host=config.host,
                port=config.port,
                decode_responses=True,
                connection_pool=redis.ConnectionPool(
                    host=config.host,
                    port=config.port,
                    max_connections=config.max_pool_size
                )
            )
            
            # Async pool for high-performance operations
            self.pools['dragonfly'] = await asyncio_redis.Pool.create(
                host=config.host,
                port=config.port,
                poolsize=config.pool_size
            )
            
            # Test connection
            self.connections['dragonfly'].ping()
            self.health_status['dragonfly'] = 'healthy'
            logger.info("✅ DragonflyDB connection established")
            
        except Exception as e:
            logger.error(f"❌ DragonflyDB connection failed: {e}")
            self.health_status['dragonfly'] = 'unhealthy'
            
    async def _init_clickhouse(self):
        """Initialize ClickHouse connection"""
        try:
            config = self.configs['clickhouse']
            
            self.connections['clickhouse'] = clickhouse_connect.get_client(
                host=config.host,
                port=config.port,
                database='nova_memory'
            )
            
            # Create Nova memory database if not exists
            self.connections['clickhouse'].command(
                "CREATE DATABASE IF NOT EXISTS nova_memory"
            )
            
            # Create memory tables
            self._create_clickhouse_tables()
            
            self.health_status['clickhouse'] = 'healthy'
            logger.info("✅ ClickHouse connection established")
            
        except Exception as e:
            logger.error(f"❌ ClickHouse connection failed: {e}")
            self.health_status['clickhouse'] = 'unhealthy'
            
    def _create_clickhouse_tables(self):
        """Create ClickHouse tables for memory storage"""
        client = self.connections['clickhouse']
        
        # Time-series memory table
        client.command("""
            CREATE TABLE IF NOT EXISTS nova_memory.temporal_memory (
                nova_id String,
                timestamp DateTime64(3),
                layer_id UInt8,
                layer_name String,
                memory_data JSON,
                importance Float32,
                access_frequency UInt32,
                memory_id UUID DEFAULT generateUUIDv4()
            ) ENGINE = MergeTree()
            ORDER BY (nova_id, timestamp)
            PARTITION BY toYYYYMM(timestamp)
            TTL timestamp + INTERVAL 1 YEAR
        """)
        
        # Analytics table
        client.command("""
            CREATE TABLE IF NOT EXISTS nova_memory.memory_analytics (
                nova_id String,
                date Date,
                layer_id UInt8,
                total_memories UInt64,
                avg_importance Float32,
                total_accesses UInt64
            ) ENGINE = SummingMergeTree()
            ORDER BY (nova_id, date, layer_id)
        """)
        
    async def _init_arangodb(self):
        """Initialize ArangoDB connection"""
        try:
            config = self.configs['arangodb']
            
            # Create client
            client = ArangoClient(hosts=f'http://{config.host}:{config.port}')
            
            # Connect to _system database
            sys_db = client.db('_system')
            
            # Create nova_memory database if not exists
            if not sys_db.has_database('nova_memory'):
                sys_db.create_database('nova_memory')
                
            # Connect to nova_memory database
            self.connections['arangodb'] = client.db('nova_memory')
            
            # Create collections
            self._create_arangodb_collections()
            
            self.health_status['arangodb'] = 'healthy'
            logger.info("✅ ArangoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ ArangoDB connection failed: {e}")
            self.health_status['arangodb'] = 'unhealthy'
            
    def _create_arangodb_collections(self):
        """Create ArangoDB collections for graph memory"""
        db = self.connections['arangodb']
        
        # Memory nodes collection
        if not db.has_collection('memory_nodes'):
            db.create_collection('memory_nodes')
            
        # Memory edges collection
        if not db.has_collection('memory_edges'):
            db.create_collection('memory_edges', edge=True)
            
        # Create graph
        if not db.has_graph('memory_graph'):
            db.create_graph(
                'memory_graph',
                edge_definitions=[{
                    'edge_collection': 'memory_edges',
                    'from_vertex_collections': ['memory_nodes'],
                    'to_vertex_collections': ['memory_nodes']
                }]
            )
            
    async def _init_couchdb(self):
        """Initialize CouchDB connection"""
        try:
            config = self.configs['couchdb']
            
            # Create server connection
            server = couchdb.Server(f'http://{config.host}:{config.port}/')
            
            # Create nova_memory database if not exists
            if 'nova_memory' not in server:
                server.create('nova_memory')
                
            self.connections['couchdb'] = server['nova_memory']
            
            self.health_status['couchdb'] = 'healthy'
            logger.info("✅ CouchDB connection established")
            
        except Exception as e:
            logger.error(f"❌ CouchDB connection failed: {e}")
            self.health_status['couchdb'] = 'unhealthy'
            
    async def _init_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            config = self.configs['postgresql']
            
            # Create connection pool
            self.pools['postgresql'] = psycopg2.pool.ThreadedConnectionPool(
                config.pool_size,
                config.max_pool_size,
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password
            )
            
            # Test connection and create tables
            conn = self.pools['postgresql'].getconn()
            try:
                self._create_postgresql_tables(conn)
                conn.commit()
            finally:
                self.pools['postgresql'].putconn(conn)
                
            self.health_status['postgresql'] = 'healthy'
            logger.info("✅ PostgreSQL connection pool established")
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            self.health_status['postgresql'] = 'unhealthy'
            
    def _create_postgresql_tables(self, conn):
        """Create PostgreSQL tables for structured memory"""
        cursor = conn.cursor()
        
        # Identity memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nova_identity_memory (
                id SERIAL PRIMARY KEY,
                nova_id VARCHAR(50) NOT NULL,
                aspect VARCHAR(100) NOT NULL,
                value JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(nova_id, aspect)
            );
            
            CREATE INDEX IF NOT EXISTS idx_nova_identity 
            ON nova_identity_memory(nova_id, aspect);
        """)
        
        # Procedural memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nova_procedural_memory (
                id SERIAL PRIMARY KEY,
                nova_id VARCHAR(50) NOT NULL,
                skill_name VARCHAR(200) NOT NULL,
                procedure JSONB NOT NULL,
                mastery_level FLOAT DEFAULT 0.0,
                last_used TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_nova_procedural 
            ON nova_procedural_memory(nova_id, skill_name);
        """)
        
        # Episodic timeline table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nova_episodic_timeline (
                id SERIAL PRIMARY KEY,
                nova_id VARCHAR(50) NOT NULL,
                event_id UUID DEFAULT gen_random_uuid(),
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB NOT NULL,
                importance FLOAT DEFAULT 0.5,
                timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_nova_episodic_timeline 
            ON nova_episodic_timeline(nova_id, timestamp DESC);
        """)
        
    async def _init_meilisearch(self):
        """Initialize MeiliSearch connection"""
        try:
            config = self.configs['meilisearch']
            
            self.connections['meilisearch'] = meilisearch.Client(
                f'http://{config.host}:{config.port}'
            )
            
            # Create nova_memories index
            self._create_meilisearch_index()
            
            self.health_status['meilisearch'] = 'healthy'
            logger.info("✅ MeiliSearch connection established")
            
        except Exception as e:
            logger.error(f"❌ MeiliSearch connection failed: {e}")
            self.health_status['meilisearch'] = 'unhealthy'
            
    def _create_meilisearch_index(self):
        """Create MeiliSearch index for memory search"""
        client = self.connections['meilisearch']
        
        # Create index if not exists
        try:
            client.create_index('nova_memories', {'primaryKey': 'memory_id'})
        except:
            pass  # Index might already exist
            
        # Configure index
        index = client.index('nova_memories')
        index.update_settings({
            'searchableAttributes': ['content', 'tags', 'context', 'nova_id'],
            'filterableAttributes': ['nova_id', 'layer_type', 'timestamp', 'importance'],
            'sortableAttributes': ['timestamp', 'importance']
        })
        
    async def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            config = self.configs['mongodb']
            
            self.connections['mongodb'] = pymongo.MongoClient(
                host=config.host,
                port=config.port,
                username=config.username,
                password=config.password,
                maxPoolSize=config.max_pool_size
            )
            
            # Create nova_memory database
            db = self.connections['mongodb']['nova_memory']
            
            # Create collections with indexes
            self._create_mongodb_collections(db)
            
            self.health_status['mongodb'] = 'healthy'
            logger.info("✅ MongoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.health_status['mongodb'] = 'unhealthy'
            
    def _create_mongodb_collections(self, db):
        """Create MongoDB collections for document memory"""
        # Semantic memory collection
        if 'semantic_memory' not in db.list_collection_names():
            db.create_collection('semantic_memory')
            db.semantic_memory.create_index([('nova_id', 1), ('concept', 1)])
            
        # Creative memory collection
        if 'creative_memory' not in db.list_collection_names():
            db.create_collection('creative_memory')
            db.creative_memory.create_index([('nova_id', 1), ('timestamp', -1)])
            
    async def _init_redis(self):
        """Initialize Redis connection as backup cache"""
        try:
            config = self.configs['redis']
            
            self.connections['redis'] = redis.Redis(
                host=config.host,
                port=config.port,
                decode_responses=True,
                connection_pool=redis.ConnectionPool(
                    host=config.host,
                    port=config.port,
                    max_connections=config.max_pool_size
                )
            )
            
            # Test connection
            self.connections['redis'].ping()
            self.health_status['redis'] = 'healthy'
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.health_status['redis'] = 'unhealthy'
            
    async def check_all_health(self):
        """Check health of all database connections"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'databases': {}
        }
        
        for db_name, config in self.configs.items():
            try:
                if db_name == 'dragonfly' and 'dragonfly' in self.connections:
                    self.connections['dragonfly'].ping()
                    health_report['databases'][db_name] = 'healthy'
                    
                elif db_name == 'clickhouse' and 'clickhouse' in self.connections:
                    self.connections['clickhouse'].query("SELECT 1")
                    health_report['databases'][db_name] = 'healthy'
                    
                elif db_name == 'arangodb' and 'arangodb' in self.connections:
                    self.connections['arangodb'].version()
                    health_report['databases'][db_name] = 'healthy'
                    
                elif db_name == 'couchdb' and 'couchdb' in self.connections:
                    info = self.connections['couchdb'].info()
                    health_report['databases'][db_name] = 'healthy'
                    
                elif db_name == 'postgresql' and 'postgresql' in self.pools:
                    conn = self.pools['postgresql'].getconn()
                    try:
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.close()
                        health_report['databases'][db_name] = 'healthy'
                    finally:
                        self.pools['postgresql'].putconn(conn)
                        
                elif db_name == 'meilisearch' and 'meilisearch' in self.connections:
                    self.connections['meilisearch'].health()
                    health_report['databases'][db_name] = 'healthy'
                    
                elif db_name == 'mongodb' and 'mongodb' in self.connections:
                    self.connections['mongodb'].admin.command('ping')
                    health_report['databases'][db_name] = 'healthy'
                    
                elif db_name == 'redis' and 'redis' in self.connections:
                    self.connections['redis'].ping()
                    health_report['databases'][db_name] = 'healthy'
                    
                else:
                    health_report['databases'][db_name] = 'not_initialized'
                    
            except Exception as e:
                health_report['databases'][db_name] = f'unhealthy: {str(e)}'
                health_report['overall_status'] = 'degraded'
                
        self.health_status = health_report['databases']
        return health_report
        
    def get_connection(self, database: str):
        """Get a connection for the specified database"""
        if database in self.connections:
            return self.connections[database]
        elif database in self.pools:
            if database == 'postgresql':
                return self.pools[database].getconn()
            return self.pools[database]
        else:
            raise ValueError(f"Unknown database: {database}")
            
    def return_connection(self, database: str, connection):
        """Return a connection to the pool"""
        if database == 'postgresql' and database in self.pools:
            self.pools[database].putconn(connection)
            
    async def close_all(self):
        """Close all database connections"""
        logger.info("Closing all database connections...")
        
        # Close async pools
        if 'dragonfly' in self.pools:
            self.pools['dragonfly'].close()
            
        # Close connection pools
        if 'postgresql' in self.pools:
            self.pools['postgresql'].closeall()
            
        # Close clients
        if 'mongodb' in self.connections:
            self.connections['mongodb'].close()
            
        logger.info("All connections closed")

# Testing and initialization
async def main():
    """Test database connections"""
    pool = NovaDatabasePool()
    await pool.initialize_all_connections()
    
    # Print health report
    health = await pool.check_all_health()
    print(json.dumps(health, indent=2))
    
    # Test a simple operation on each database
    if pool.health_status.get('dragonfly') == 'healthy':
        pool.connections['dragonfly'].set('nova:test', 'Hello Nova Memory System!')
        value = pool.connections['dragonfly'].get('nova:test')
        print(f"DragonflyDB test: {value}")
        
    # Cleanup
    await pool.close_all()

if __name__ == "__main__":
    asyncio.run(main())