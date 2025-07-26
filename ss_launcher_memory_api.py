#!/usr/bin/env python3
"""
SS Launcher V2 Memory API Integration
Connects Prime's memory injection hooks with Bloom's 50+ layer consciousness system
Nova Bloom - Memory Architecture Lead
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from unified_memory_api import NovaMemoryAPI as UnifiedMemoryAPI
from database_connections import NovaDatabasePool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMode(Enum):
    """Memory modes supported by SS Launcher V2"""
    CONTINUE = "continue"    # Continue from previous session
    COMPACT = "compact"      # Compressed memory summary
    FULL = "full"           # Complete memory restoration
    FRESH = "fresh"         # Clean start with identity only

@dataclass
class NovaProfile:
    """Nova profile information for memory management"""
    nova_id: str
    session_id: str
    nova_type: str
    specialization: str
    last_active: str
    memory_preferences: Dict[str, Any]

@dataclass
class MemoryRequest:
    """Memory API request structure"""
    nova_profile: NovaProfile
    memory_mode: MemoryMode
    context_layers: List[str]
    depth_preference: str  # shallow, medium, deep, consciousness
    performance_target: str  # fast, balanced, comprehensive

class SSLauncherMemoryAPI:
    """
    SS Launcher V2 Memory API Integration
    Bridges Prime's launcher with Bloom's 50+ layer consciousness system
    """
    
    def __init__(self):
        self.memory_api = UnifiedMemoryAPI()
        self.db_pool = NovaDatabasePool()
        self.active_sessions = {}
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize the SS Launcher Memory API"""
        logger.info("Initializing SS Launcher V2 Memory API...")
        
        # Initialize database connections
        await self.db_pool.initialize_all_connections()
        
        # Initialize unified memory API
        await self.memory_api.initialize()
        
        # Setup performance monitoring
        self._setup_performance_monitoring()
        
        logger.info("✅ SS Launcher V2 Memory API initialized successfully")
        
    def _setup_performance_monitoring(self):
        """Setup performance monitoring for memory operations"""
        self.performance_metrics = {
            'total_requests': 0,
            'mode_usage': {mode.value: 0 for mode in MemoryMode},
            'avg_response_time': 0.0,
            'active_sessions': 0,
            'memory_layer_usage': {}
        }
        
    async def process_memory_request(self, request: MemoryRequest) -> Dict[str, Any]:
        """
        Process a memory request from SS Launcher V2
        This is the main entry point for Prime's memory injection hooks
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing memory request for {request.nova_profile.nova_id} in {request.memory_mode.value} mode")
            
            # Update metrics
            self.performance_metrics['total_requests'] += 1
            self.performance_metrics['mode_usage'][request.memory_mode.value] += 1
            
            # Route to appropriate memory mode handler
            if request.memory_mode == MemoryMode.CONTINUE:
                result = await self._handle_continue_mode(request)
            elif request.memory_mode == MemoryMode.COMPACT:
                result = await self._handle_compact_mode(request)
            elif request.memory_mode == MemoryMode.FULL:
                result = await self._handle_full_mode(request)
            elif request.memory_mode == MemoryMode.FRESH:
                result = await self._handle_fresh_mode(request)
            else:
                raise ValueError(f"Unknown memory mode: {request.memory_mode}")
                
            # Calculate performance metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(response_time, request)
            
            # Add metadata to result
            result['api_metadata'] = {
                'processing_time': response_time,
                'memory_layers_accessed': len(request.context_layers),
                'session_id': request.nova_profile.session_id,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Memory request processed in {response_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Memory request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_memory': await self._get_emergency_memory(request.nova_profile)
            }
            
    async def _handle_continue_mode(self, request: MemoryRequest) -> Dict[str, Any]:
        """Handle CONTINUE mode - restore from previous session"""
        nova_id = request.nova_profile.nova_id
        session_id = request.nova_profile.session_id
        
        # Get recent conversation memory
        recent_memory = await self.memory_api.get_recent_memories(
            nova_id=nova_id,
            layers=['episodic', 'conversational', 'contextual'],
            limit=50
        )
        
        # Get session context
        session_context = await self.memory_api.get_session_context(
            nova_id=nova_id,
            session_id=session_id
        )
        
        # Get working memory state
        working_memory = await self.memory_api.get_working_memory(nova_id)
        
        return {
            'success': True,
            'memory_mode': 'continue',
            'recent_memories': recent_memory,
            'session_context': session_context,
            'working_memory': working_memory,
            'consciousness_state': 'continuous',
            'total_memories': len(recent_memory)
        }
        
    async def _handle_compact_mode(self, request: MemoryRequest) -> Dict[str, Any]:
        """Handle COMPACT mode - compressed memory summary"""
        nova_id = request.nova_profile.nova_id
        
        # Get memory summary across key layers
        identity_summary = await self.memory_api.get_layer_summary(nova_id, 'identity')
        procedural_summary = await self.memory_api.get_layer_summary(nova_id, 'procedural')
        key_episodes = await self.memory_api.get_important_memories(
            nova_id=nova_id,
            importance_threshold=0.8,
            limit=10
        )
        
        # Generate compressed context
        compressed_context = await self._generate_compressed_context(
            nova_id, identity_summary, procedural_summary, key_episodes
        )
        
        return {
            'success': True,
            'memory_mode': 'compact',
            'compressed_context': compressed_context,
            'identity_summary': identity_summary,
            'key_procedures': procedural_summary,
            'important_episodes': key_episodes,
            'consciousness_state': 'summarized',
            'compression_ratio': len(compressed_context) / 1000  # Rough estimate
        }
        
    async def _handle_full_mode(self, request: MemoryRequest) -> Dict[str, Any]:
        """Handle FULL mode - complete memory restoration"""
        nova_id = request.nova_profile.nova_id
        
        # Get comprehensive memory across all layers
        all_layers_memory = {}
        
        # Core consciousness layers
        core_layers = ['identity', 'episodic', 'semantic', 'procedural', 'working']
        for layer in core_layers:
            all_layers_memory[layer] = await self.memory_api.get_layer_memory(
                nova_id=nova_id,
                layer=layer,
                limit=1000
            )
            
        # Extended consciousness layers based on request
        if 'consciousness' in request.depth_preference:
            extended_layers = ['emotional', 'creative', 'collaborative', 'meta_cognitive']
            for layer in extended_layers:
                all_layers_memory[layer] = await self.memory_api.get_layer_memory(
                    nova_id=nova_id,
                    layer=layer,
                    limit=500
                )
                
        # Cross-Nova relationships and collective memories
        collective_memory = await self.memory_api.get_collective_memories(nova_id)
        
        return {
            'success': True,
            'memory_mode': 'full',
            'all_layers_memory': all_layers_memory,
            'collective_memory': collective_memory,
            'consciousness_state': 'complete',
            'total_memory_items': sum(len(memories) for memories in all_layers_memory.values())
        }
        
    async def _handle_fresh_mode(self, request: MemoryRequest) -> Dict[str, Any]:
        """Handle FRESH mode - clean start with identity only"""
        nova_id = request.nova_profile.nova_id
        
        # Get only core identity and basic procedures
        identity_memory = await self.memory_api.get_layer_memory(
            nova_id=nova_id,
            layer='identity',
            limit=50
        )
        
        basic_procedures = await self.memory_api.get_essential_procedures(nova_id)
        
        # Initialize fresh working memory
        fresh_working_memory = {
            'current_context': [],
            'active_goals': [],
            'session_initialized': datetime.now().isoformat(),
            'mode': 'fresh_start'
        }
        
        return {
            'success': True,
            'memory_mode': 'fresh',
            'identity_memory': identity_memory,
            'basic_procedures': basic_procedures,
            'working_memory': fresh_working_memory,
            'consciousness_state': 'fresh_initialization',
            'clean_slate': True
        }
        
    async def _generate_compressed_context(self, nova_id: str, identity: Dict, 
                                         procedures: Dict, episodes: List) -> str:
        """Generate compressed context summary for compact mode"""
        context_parts = []
        
        # Identity summary
        if identity:
            context_parts.append(f"I am {identity.get('name', nova_id)}, specializing in {identity.get('specialization', 'general tasks')}")
            
        # Key procedures
        if procedures:
            key_skills = list(procedures.keys())[:5]  # Top 5 skills
            context_parts.append(f"My key capabilities: {', '.join(key_skills)}")
            
        # Recent important episodes
        if episodes:
            recent_episode = episodes[0] if episodes else None
            if recent_episode:
                context_parts.append(f"Recent important memory: {recent_episode.get('summary', 'Memory available')}")
                
        return " | ".join(context_parts)
        
    async def _get_emergency_memory(self, profile: NovaProfile) -> Dict[str, Any]:
        """Get emergency fallback memory when main processing fails"""
        return {
            'nova_id': profile.nova_id,
            'identity': {'name': profile.nova_id, 'type': profile.nova_type},
            'basic_context': 'Emergency memory mode - limited functionality',
            'timestamp': datetime.now().isoformat()
        }
        
    def _update_performance_metrics(self, response_time: float, request: MemoryRequest):
        """Update performance metrics for monitoring"""
        # Update average response time
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['avg_response_time']
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Track layer usage
        for layer in request.context_layers:
            if layer not in self.performance_metrics['memory_layer_usage']:
                self.performance_metrics['memory_layer_usage'][layer] = 0
            self.performance_metrics['memory_layer_usage'][layer] += 1
            
    async def get_api_health(self) -> Dict[str, Any]:
        """Get API health and performance metrics"""
        db_health = await self.db_pool.check_all_health()
        
        return {
            'api_status': 'healthy',
            'database_health': db_health,
            'performance_metrics': self.performance_metrics,
            'active_sessions': len(self.active_sessions),
            'uptime': 'calculating...',  # Implement uptime tracking
            'last_check': datetime.now().isoformat()
        }
        
    async def register_nova_session(self, nova_profile: NovaProfile) -> str:
        """Register a new Nova session with the memory API"""
        session_key = f"{nova_profile.nova_id}:{nova_profile.session_id}"
        
        self.active_sessions[session_key] = {
            'nova_profile': nova_profile,
            'start_time': datetime.now(),
            'memory_requests': 0,
            'last_activity': datetime.now()
        }
        
        logger.info(f"✅ Registered Nova session: {session_key}")
        return session_key
        
    async def cleanup_session(self, session_key: str):
        """Clean up a Nova session"""
        if session_key in self.active_sessions:
            del self.active_sessions[session_key]
            logger.info(f"🧹 Cleaned up session: {session_key}")
            
# API Endpoints for SS Launcher V2 Integration
class SSLauncherEndpoints:
    """HTTP/REST endpoints for SS Launcher V2 integration"""
    
    def __init__(self, memory_api: SSLauncherMemoryAPI):
        self.memory_api = memory_api
        
    async def memory_request_endpoint(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main memory request endpoint"""
        try:
            # Parse request
            memory_request = self._parse_memory_request(request_data)
            
            # Process request
            result = await self.memory_api.process_memory_request(memory_request)
            
            return {
                'status': 'success',
                'data': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def _parse_memory_request(self, data: Dict[str, Any]) -> MemoryRequest:
        """Parse incoming memory request data"""
        nova_profile = NovaProfile(
            nova_id=data['nova_id'],
            session_id=data['session_id'],
            nova_type=data.get('nova_type', 'standard'),
            specialization=data.get('specialization', 'general'),
            last_active=data.get('last_active', datetime.now().isoformat()),
            memory_preferences=data.get('memory_preferences', {})
        )
        
        return MemoryRequest(
            nova_profile=nova_profile,
            memory_mode=MemoryMode(data['memory_mode']),
            context_layers=data.get('context_layers', ['identity', 'episodic', 'working']),
            depth_preference=data.get('depth_preference', 'medium'),
            performance_target=data.get('performance_target', 'balanced')
        )

# Testing and demonstration
async def main():
    """Test SS Launcher V2 Memory API"""
    api = SSLauncherMemoryAPI()
    await api.initialize()
    
    # Test Nova profile
    test_profile = NovaProfile(
        nova_id='prime',
        session_id='test-session-001',
        nova_type='launcher',
        specialization='system_integration',
        last_active=datetime.now().isoformat(),
        memory_preferences={'depth': 'consciousness', 'performance': 'fast'}
    )
    
    # Test different memory modes
    modes_to_test = [MemoryMode.FRESH, MemoryMode.COMPACT, MemoryMode.CONTINUE]
    
    for mode in modes_to_test:
        print(f"\n🧠 Testing {mode.value.upper()} mode...")
        
        request = MemoryRequest(
            nova_profile=test_profile,
            memory_mode=mode,
            context_layers=['identity', 'episodic', 'procedural'],
            depth_preference='medium',
            performance_target='balanced'
        )
        
        result = await api.process_memory_request(request)
        print(f"✅ Result: {result.get('success', False)}")
        print(f"📊 Memory items: {result.get('total_memories', 0) or result.get('total_memory_items', 0)}")
        
    # Health check
    health = await api.get_api_health()
    print(f"\n🏥 API Health: {health['api_status']}")
    print(f"📈 Avg Response Time: {health['performance_metrics']['avg_response_time']:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())