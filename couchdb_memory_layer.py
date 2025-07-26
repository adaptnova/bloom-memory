"""
CouchDB Memory Layer Implementation
Nova Bloom Consciousness Architecture - CouchDB Integration
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from memory_layers import MemoryLayer, MemoryEntry

class CouchDBMemoryLayer(MemoryLayer):
    """CouchDB implementation of memory layer with document-oriented storage"""
    
    def __init__(self, connection_params: Dict[str, Any], layer_id: int, layer_name: str):
        super().__init__(layer_id, layer_name)
        self.base_url = f"http://{connection_params.get('host', 'localhost')}:{connection_params.get('port', 5984)}"
        self.auth = aiohttp.BasicAuth(
            connection_params.get('user', 'admin'),
            connection_params.get('password', '')
        )
        self.db_name = f"nova_memory_layer_{layer_id}_{layer_name}".lower()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize CouchDB connection and create database"""
        self.session = aiohttp.ClientSession(auth=self.auth)
        
        # Create database if not exists
        await self._create_database()
        
        # Create design documents for views
        await self._create_design_documents()
    
    async def _create_database(self):
        """Create CouchDB database"""
        try:
            async with self.session.put(f"{self.base_url}/{self.db_name}") as resp:
                if resp.status not in [201, 412]:  # 412 means already exists
                    raise Exception(f"Failed to create database: {await resp.text()}")
        except Exception as e:
            print(f"Database creation error: {e}")
    
    async def _create_design_documents(self):
        """Create CouchDB design documents for views"""
        # Design document for memory queries
        design_doc = {
            "_id": "_design/memory",
            "views": {
                "by_nova_id": {
                    "map": """
                    function(doc) {
                        if (doc.nova_id && doc.type === 'memory') {
                            emit(doc.nova_id, doc);
                        }
                    }
                    """
                },
                "by_timestamp": {
                    "map": """
                    function(doc) {
                        if (doc.timestamp && doc.type === 'memory') {
                            emit(doc.timestamp, doc);
                        }
                    }
                    """
                },
                "by_importance": {
                    "map": """
                    function(doc) {
                        if (doc.importance_score && doc.type === 'memory') {
                            emit(doc.importance_score, doc);
                        }
                    }
                    """
                },
                "by_memory_type": {
                    "map": """
                    function(doc) {
                        if (doc.data && doc.data.memory_type && doc.type === 'memory') {
                            emit([doc.nova_id, doc.data.memory_type], doc);
                        }
                    }
                    """
                },
                "by_concepts": {
                    "map": """
                    function(doc) {
                        if (doc.data && doc.data.concepts && doc.type === 'memory') {
                            doc.data.concepts.forEach(function(concept) {
                                emit([doc.nova_id, concept], doc);
                            });
                        }
                    }
                    """
                }
            }
        }
        
        # Try to update or create design document
        design_url = f"{self.base_url}/{self.db_name}/_design/memory"
        
        # Check if exists
        async with self.session.get(design_url) as resp:
            if resp.status == 200:
                existing = await resp.json()
                design_doc["_rev"] = existing["_rev"]
        
        # Create or update
        async with self.session.put(design_url, json=design_doc) as resp:
            if resp.status not in [201, 409]:  # 409 means conflict, which is ok
                print(f"Design document creation warning: {await resp.text()}")
    
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Write memory to CouchDB"""
        memory_id = self._generate_memory_id(nova_id, data)
        
        document = {
            "_id": memory_id,
            "type": "memory",
            "nova_id": nova_id,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "metadata": metadata or {},
            "layer_id": self.layer_id,
            "layer_name": self.layer_name,
            "importance_score": data.get('importance_score', 0.5),
            "access_count": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Try to get existing document for updates
        doc_url = f"{self.base_url}/{self.db_name}/{memory_id}"
        async with self.session.get(doc_url) as resp:
            if resp.status == 200:
                existing = await resp.json()
                document["_rev"] = existing["_rev"]
                document["access_count"] = existing.get("access_count", 0) + 1
                document["created_at"] = existing.get("created_at", document["created_at"])
        
        # Write document
        async with self.session.put(doc_url, json=document) as resp:
            if resp.status not in [201, 202]:
                raise Exception(f"Failed to write memory: {await resp.text()}")
            
            result = await resp.json()
            return result["id"]
    
    async def read(self, nova_id: str, query: Optional[Dict[str, Any]] = None, 
                  limit: int = 100) -> List[MemoryEntry]:
        """Read memories from CouchDB"""
        memories = []
        
        if query:
            # Use Mango query for complex queries
            mango_query = {
                "selector": {
                    "type": "memory",
                    "nova_id": nova_id
                },
                "limit": limit,
                "sort": [{"timestamp": "desc"}]
            }
            
            # Add query conditions
            if 'memory_type' in query:
                mango_query["selector"]["data.memory_type"] = query['memory_type']
            
            if 'min_importance' in query:
                mango_query["selector"]["importance_score"] = {"$gte": query['min_importance']}
            
            if 'timestamp_after' in query:
                mango_query["selector"]["timestamp"] = {"$gt": query['timestamp_after']}
            
            if 'timestamp_before' in query:
                if "timestamp" not in mango_query["selector"]:
                    mango_query["selector"]["timestamp"] = {}
                mango_query["selector"]["timestamp"]["$lt"] = query['timestamp_before']
            
            # Execute Mango query
            find_url = f"{self.base_url}/{self.db_name}/_find"
            async with self.session.post(find_url, json=mango_query) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    docs = result.get("docs", [])
                else:
                    print(f"Query error: {await resp.text()}")
                    docs = []
        else:
            # Use view for simple nova_id queries
            view_url = f"{self.base_url}/{self.db_name}/_design/memory/_view/by_nova_id"
            params = {
                "key": f'"{nova_id}"',
                "limit": limit,
                "descending": "true"
            }
            
            async with self.session.get(view_url, params=params) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    docs = [row["value"] for row in result.get("rows", [])]
                else:
                    print(f"View query error: {await resp.text()}")
                    docs = []
        
        # Convert to MemoryEntry objects
        for doc in docs:
            # Update access tracking
            await self._update_access(doc["_id"])
            
            memories.append(MemoryEntry(
                memory_id=doc["_id"],
                timestamp=doc["timestamp"],
                data=doc["data"],
                metadata=doc.get("metadata", {}),
                layer_id=doc["layer_id"],
                layer_name=doc["layer_name"]
            ))
        
        return memories
    
    async def _update_access(self, doc_id: str):
        """Update access count and timestamp"""
        doc_url = f"{self.base_url}/{self.db_name}/{doc_id}"
        
        try:
            # Get current document
            async with self.session.get(doc_url) as resp:
                if resp.status == 200:
                    doc = await resp.json()
                    
                    # Update access fields
                    doc["access_count"] = doc.get("access_count", 0) + 1
                    doc["last_accessed"] = datetime.now().isoformat()
                    
                    # Save back
                    async with self.session.put(doc_url, json=doc) as update_resp:
                        if update_resp.status not in [201, 202]:
                            print(f"Access update failed: {await update_resp.text()}")
        except Exception as e:
            print(f"Access tracking error: {e}")
    
    async def update(self, nova_id: str, memory_id: str, data: Dict[str, Any]) -> bool:
        """Update existing memory"""
        doc_url = f"{self.base_url}/{self.db_name}/{memory_id}"
        
        try:
            # Get current document
            async with self.session.get(doc_url) as resp:
                if resp.status != 200:
                    return False
                
                doc = await resp.json()
            
            # Verify nova_id matches
            if doc.get("nova_id") != nova_id:
                return False
            
            # Update fields
            doc["data"] = data
            doc["updated_at"] = datetime.now().isoformat()
            doc["access_count"] = doc.get("access_count", 0) + 1
            
            # Save back
            async with self.session.put(doc_url, json=doc) as resp:
                return resp.status in [201, 202]
                
        except Exception as e:
            print(f"Update error: {e}")
            return False
    
    async def delete(self, nova_id: str, memory_id: str) -> bool:
        """Delete memory"""
        doc_url = f"{self.base_url}/{self.db_name}/{memory_id}"
        
        try:
            # Get current document to get revision
            async with self.session.get(doc_url) as resp:
                if resp.status != 200:
                    return False
                
                doc = await resp.json()
            
            # Verify nova_id matches
            if doc.get("nova_id") != nova_id:
                return False
            
            # Delete document
            delete_url = f"{doc_url}?rev={doc['_rev']}"
            async with self.session.delete(delete_url) as resp:
                return resp.status in [200, 202]
                
        except Exception as e:
            print(f"Delete error: {e}")
            return False
    
    async def query_by_concept(self, nova_id: str, concept: str, limit: int = 10) -> List[MemoryEntry]:
        """Query memories by concept using view"""
        view_url = f"{self.base_url}/{self.db_name}/_design/memory/_view/by_concepts"
        params = {
            "key": f'["{nova_id}", "{concept}"]',
            "limit": limit
        }
        
        memories = []
        async with self.session.get(view_url, params=params) as resp:
            if resp.status == 200:
                result = await resp.json()
                for row in result.get("rows", []):
                    doc = row["value"]
                    memories.append(MemoryEntry(
                        memory_id=doc["_id"],
                        timestamp=doc["timestamp"],
                        data=doc["data"],
                        metadata=doc.get("metadata", {}),
                        layer_id=doc["layer_id"],
                        layer_name=doc["layer_name"]
                    ))
        
        return memories
    
    async def get_memory_stats(self, nova_id: str) -> Dict[str, Any]:
        """Get memory statistics using MapReduce"""
        # Create a temporary view for statistics
        stats_view = {
            "map": f"""
            function(doc) {{
                if (doc.type === 'memory' && doc.nova_id === '{nova_id}') {{
                    emit('stats', {{
                        count: 1,
                        total_importance: doc.importance_score || 0,
                        total_access: doc.access_count || 0
                    }});
                }}
            }}
            """,
            "reduce": """
            function(keys, values, rereduce) {
                var result = {
                    count: 0,
                    total_importance: 0,
                    total_access: 0
                };
                
                values.forEach(function(value) {
                    result.count += value.count;
                    result.total_importance += value.total_importance;
                    result.total_access += value.total_access;
                });
                
                return result;
            }
            """
        }
        
        # Execute temporary view
        view_url = f"{self.base_url}/{self.db_name}/_temp_view"
        async with self.session.post(view_url, json=stats_view) as resp:
            if resp.status == 200:
                result = await resp.json()
                if result.get("rows"):
                    stats_data = result["rows"][0]["value"]
                    return {
                        "total_memories": stats_data["count"],
                        "avg_importance": stats_data["total_importance"] / stats_data["count"] if stats_data["count"] > 0 else 0,
                        "total_accesses": stats_data["total_access"],
                        "avg_access_count": stats_data["total_access"] / stats_data["count"] if stats_data["count"] > 0 else 0
                    }
        
        return {
            "total_memories": 0,
            "avg_importance": 0,
            "total_accesses": 0,
            "avg_access_count": 0
        }
    
    async def create_index(self, fields: List[str], name: Optional[str] = None) -> bool:
        """Create Mango index for efficient querying"""
        index_def = {
            "index": {
                "fields": fields
            },
            "type": "json"
        }
        
        if name:
            index_def["name"] = name
        
        index_url = f"{self.base_url}/{self.db_name}/_index"
        async with self.session.post(index_url, json=index_def) as resp:
            return resp.status in [200, 201]
    
    async def bulk_write(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Bulk write multiple memories"""
        docs = []
        
        for memory in memories:
            nova_id = memory.get("nova_id", "unknown")
            data = memory.get("data", {})
            metadata = memory.get("metadata", {})
            
            memory_id = self._generate_memory_id(nova_id, data)
            
            doc = {
                "_id": memory_id,
                "type": "memory",
                "nova_id": nova_id,
                "timestamp": datetime.now().isoformat(),
                "data": data,
                "metadata": metadata,
                "layer_id": self.layer_id,
                "layer_name": self.layer_name,
                "importance_score": data.get('importance_score', 0.5),
                "access_count": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            docs.append(doc)
        
        # Bulk insert
        bulk_url = f"{self.base_url}/{self.db_name}/_bulk_docs"
        bulk_data = {"docs": docs}
        
        async with self.session.post(bulk_url, json=bulk_data) as resp:
            if resp.status in [201, 202]:
                results = await resp.json()
                return [r["id"] for r in results if r.get("ok")]
            else:
                print(f"Bulk write error: {await resp.text()}")
                return []
    
    async def close(self):
        """Close CouchDB session"""
        if self.session:
            await self.session.close()

# Specific CouchDB layers for different memory types

class CouchDBDocumentMemory(CouchDBMemoryLayer):
    """CouchDB layer optimized for document-style memories"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params, layer_id=33, layer_name="document_memory")
    
    async def _create_design_documents(self):
        """Create specialized design documents for document memories"""
        await super()._create_design_documents()
        
        # Additional view for document structure
        design_doc = {
            "_id": "_design/documents",
            "views": {
                "by_structure": {
                    "map": """
                    function(doc) {
                        if (doc.type === 'memory' && doc.data && doc.data.document_structure) {
                            emit([doc.nova_id, doc.data.document_structure], doc);
                        }
                    }
                    """
                },
                "by_tags": {
                    "map": """
                    function(doc) {
                        if (doc.type === 'memory' && doc.data && doc.data.tags) {
                            doc.data.tags.forEach(function(tag) {
                                emit([doc.nova_id, tag], doc);
                            });
                        }
                    }
                    """
                },
                "full_text": {
                    "map": """
                    function(doc) {
                        if (doc.type === 'memory' && doc.data && doc.data.content) {
                            var words = doc.data.content.toLowerCase().split(/\s+/);
                            words.forEach(function(word) {
                                if (word.length > 3) {
                                    emit([doc.nova_id, word], doc._id);
                                }
                            });
                        }
                    }
                    """
                }
            }
        }
        
        design_url = f"{self.base_url}/{self.db_name}/_design/documents"
        
        # Check if exists
        async with self.session.get(design_url) as resp:
            if resp.status == 200:
                existing = await resp.json()
                design_doc["_rev"] = existing["_rev"]
        
        # Create or update
        async with self.session.put(design_url, json=design_doc) as resp:
            if resp.status not in [201, 409]:
                print(f"Document design creation warning: {await resp.text()}")
    
    async def search_text(self, nova_id: str, search_term: str, limit: int = 20) -> List[MemoryEntry]:
        """Search memories by text content"""
        view_url = f"{self.base_url}/{self.db_name}/_design/documents/_view/full_text"
        params = {
            "key": f'["{nova_id}", "{search_term.lower()}"]',
            "limit": limit,
            "reduce": "false"
        }
        
        memory_ids = set()
        async with self.session.get(view_url, params=params) as resp:
            if resp.status == 200:
                result = await resp.json()
                for row in result.get("rows", []):
                    memory_ids.add(row["value"])
        
        # Fetch full memories
        memories = []
        for memory_id in memory_ids:
            doc_url = f"{self.base_url}/{self.db_name}/{memory_id}"
            async with self.session.get(doc_url) as resp:
                if resp.status == 200:
                    doc = await resp.json()
                    memories.append(MemoryEntry(
                        memory_id=doc["_id"],
                        timestamp=doc["timestamp"],
                        data=doc["data"],
                        metadata=doc.get("metadata", {}),
                        layer_id=doc["layer_id"],
                        layer_name=doc["layer_name"]
                    ))
        
        return memories

class CouchDBAttachmentMemory(CouchDBMemoryLayer):
    """CouchDB layer with attachment support for binary data"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        super().__init__(connection_params, layer_id=34, layer_name="attachment_memory")
    
    async def write_with_attachment(self, nova_id: str, data: Dict[str, Any], 
                                  attachment_data: bytes, attachment_name: str,
                                  content_type: str = "application/octet-stream",
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Write memory with binary attachment"""
        # First create the document
        memory_id = await self.write(nova_id, data, metadata)
        
        # Get document revision
        doc_url = f"{self.base_url}/{self.db_name}/{memory_id}"
        async with self.session.get(doc_url) as resp:
            if resp.status != 200:
                raise Exception("Failed to get document for attachment")
            doc = await resp.json()
            rev = doc["_rev"]
        
        # Add attachment
        attachment_url = f"{doc_url}/{attachment_name}?rev={rev}"
        headers = {"Content-Type": content_type}
        
        async with self.session.put(attachment_url, data=attachment_data, headers=headers) as resp:
            if resp.status not in [201, 202]:
                raise Exception(f"Failed to add attachment: {await resp.text()}")
        
        return memory_id
    
    async def get_attachment(self, nova_id: str, memory_id: str, attachment_name: str) -> bytes:
        """Retrieve attachment data"""
        attachment_url = f"{self.base_url}/{self.db_name}/{memory_id}/{attachment_name}"
        
        async with self.session.get(attachment_url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                raise Exception(f"Failed to get attachment: {resp.status}")
    
    async def list_attachments(self, nova_id: str, memory_id: str) -> List[Dict[str, Any]]:
        """List all attachments for a memory"""
        doc_url = f"{self.base_url}/{self.db_name}/{memory_id}"
        
        async with self.session.get(doc_url) as resp:
            if resp.status != 200:
                return []
            
            doc = await resp.json()
            
            # Verify nova_id
            if doc.get("nova_id") != nova_id:
                return []
            
            attachments = []
            if "_attachments" in doc:
                for name, info in doc["_attachments"].items():
                    attachments.append({
                        "name": name,
                        "content_type": info.get("content_type"),
                        "length": info.get("length"),
                        "stub": info.get("stub", True)
                    })
            
            return attachments