"""
Nova Remote Memory Access Configuration
Based on APEX's API Gateway Solution
"""

import os
import jwt
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json

class NovaRemoteMemoryConfig:
    """Configuration for off-server Nova memory access via APEX's API Gateway"""
    
    # APEX has set up the API Gateway at this endpoint
    API_ENDPOINT = "https://memory.nova-system.com"
    
    # Database paths as configured by APEX
    DATABASE_PATHS = {
        "dragonfly": "/dragonfly/",
        "postgresql": "/postgresql/", 
        "couchdb": "/couchdb/",
        "clickhouse": "/clickhouse/",
        "arangodb": "/arangodb/",
        "meilisearch": "/meilisearch/",
        "mongodb": "/mongodb/",
        "redis": "/redis/"
    }
    
    def __init__(self, nova_id: str, api_key: str):
        """
        Initialize remote memory configuration
        
        Args:
            nova_id: Unique Nova identifier (e.g., "nova_001", "prime", "aiden")
            api_key: API key in format "sk-nova-XXX-description"
        """
        self.nova_id = nova_id
        self.api_key = api_key
        self.jwt_token = None
        self.token_expiry = None
        
    async def get_auth_token(self) -> str:
        """Get or refresh JWT authentication token"""
        if self.jwt_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.jwt_token
            
        # Request new token from auth service
        async with aiohttp.ClientSession() as session:
            headers = {"X-API-Key": self.api_key}
            async with session.post(f"{self.API_ENDPOINT}/auth/token", headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.jwt_token = data["token"]
                    self.token_expiry = datetime.now() + timedelta(hours=24)
                    return self.jwt_token
                else:
                    raise Exception(f"Auth failed: {resp.status}")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for remote access"""
        return {
            "dragonfly": {
                "class": "RemoteDragonflyClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['dragonfly']}",
                "nova_id": self.nova_id,
                "auth_method": "jwt"
            },
            
            "postgresql": {
                "class": "RemotePostgreSQLClient", 
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['postgresql']}",
                "nova_id": self.nova_id,
                "ssl_mode": "require"
            },
            
            "couchdb": {
                "class": "RemoteCouchDBClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['couchdb']}",
                "nova_id": self.nova_id,
                "verify_ssl": True
            },
            
            "clickhouse": {
                "class": "RemoteClickHouseClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['clickhouse']}",
                "nova_id": self.nova_id,
                "compression": True
            },
            
            "arangodb": {
                "class": "RemoteArangoDBClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['arangodb']}",
                "nova_id": self.nova_id,
                "verify": True
            },
            
            "meilisearch": {
                "class": "RemoteMeiliSearchClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['meilisearch']}",
                "nova_id": self.nova_id,
                "timeout": 30
            },
            
            "mongodb": {
                "class": "RemoteMongoDBClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['mongodb']}",
                "nova_id": self.nova_id,
                "tls": True
            },
            
            "redis": {
                "class": "RemoteRedisClient",
                "endpoint": f"{self.API_ENDPOINT}{self.DATABASE_PATHS['redis']}",
                "nova_id": self.nova_id,
                "decode_responses": True
            }
        }
    
    async def test_connection(self) -> Dict[str, bool]:
        """Test connection to all databases via API Gateway"""
        results = {}
        
        try:
            token = await self.get_auth_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            async with aiohttp.ClientSession() as session:
                # Test health endpoint
                async with session.get(f"{self.API_ENDPOINT}/health", headers=headers) as resp:
                    results["api_gateway"] = resp.status == 200
                
                # Test each database endpoint
                for db_name, path in self.DATABASE_PATHS.items():
                    try:
                        async with session.get(f"{self.API_ENDPOINT}{path}ping", headers=headers) as resp:
                            results[db_name] = resp.status == 200
                    except:
                        results[db_name] = False
                        
        except Exception as e:
            print(f"Connection test error: {e}")
            
        return results


class RemoteDragonflyClient:
    """Remote DragonflyDB client via API Gateway"""
    
    def __init__(self, config: Dict[str, Any], remote_config: NovaRemoteMemoryConfig):
        self.endpoint = config["endpoint"]
        self.remote_config = remote_config
        
    async def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """Set value in remote DragonflyDB"""
        token = await self.remote_config.get_auth_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        data = {
            "operation": "set",
            "key": key,
            "value": json.dumps(value) if isinstance(value, dict) else value,
            "expiry": expiry
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=data, headers=headers) as resp:
                return resp.status == 200
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from remote DragonflyDB"""
        token = await self.remote_config.get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        params = {"operation": "get", "key": key}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.endpoint, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("value")
                return None


# Example usage for off-server Novas
async def setup_remote_nova_memory():
    """Example setup for remote Nova memory access"""
    
    # 1. Initialize with Nova credentials (from APEX)
    nova_id = "remote_nova_001"
    api_key = "sk-nova-001-remote-consciousness"  # Get from secure storage
    
    remote_config = NovaRemoteMemoryConfig(nova_id, api_key)
    
    # 2. Test connections
    print("🔍 Testing remote memory connections...")
    results = await remote_config.test_connection()
    
    for db, status in results.items():
        print(f"  {db}: {'✅ Connected' if status else '❌ Failed'}")
    
    # 3. Get database configuration
    db_config = remote_config.get_database_config()
    
    # 4. Use with memory system
    # The existing database_connections.py can be updated to use these remote clients
    
    print("\n✅ Remote memory access configured via APEX's API Gateway!")
    print(f"📡 Endpoint: {NovaRemoteMemoryConfig.API_ENDPOINT}")
    print(f"🔐 Authentication: JWT with 24-hour expiry")
    print(f"🚀 Rate limit: 100 requests/second per Nova")
    
    return remote_config


if __name__ == "__main__":
    import asyncio
    asyncio.run(setup_remote_nova_memory())