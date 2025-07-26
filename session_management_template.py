#!/usr/bin/env python3
"""
Nova Session Management Template
Complete implementation for session state capture, persistence, and transfer
Shared by Nova Bloom for Prime's SS Launcher V2 integration
"""

import json
import asyncio
import redis
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pickle
import base64

# Database connections
DRAGONFLY_HOST = 'localhost'
DRAGONFLY_PORT = 18000
DRAGONFLY_PASSWORD = 'dragonfly-password-f7e6d5c4b3a2f1e0d9c8b7a6f5e4d3c2'

@dataclass
class SessionState:
    """Complete session state for a Nova"""
    nova_id: str
    session_id: str
    start_time: str
    last_activity: str
    working_memory: Dict[str, Any]
    context_stack: List[Dict[str, Any]]
    active_goals: List[str]
    conversation_history: List[Dict[str, Any]]
    emotional_state: Dict[str, float]
    memory_references: List[str]
    metadata: Dict[str, Any]

@dataclass
class NovaProfile:
    """Nova profile for session initialization"""
    nova_id: str
    nova_type: str
    specialization: str
    identity_traits: Dict[str, Any]
    core_procedures: List[str]
    relationship_map: Dict[str, str]
    preferences: Dict[str, Any]
    
class SessionManager:
    """
    Complete session management implementation
    Handles capture, persistence, transfer, and restoration
    """
    
    def __init__(self):
        # Initialize DragonflyDB connection
        self.redis_client = redis.Redis(
            host=DRAGONFLY_HOST,
            port=DRAGONFLY_PORT,
            password=DRAGONFLY_PASSWORD,
            decode_responses=True
        )
        
        # Session tracking
        self.active_sessions = {}
        self.session_checkpoints = {}
        
    def create_session(self, nova_profile: NovaProfile) -> SessionState:
        """Create a new session from a Nova profile"""
        session_id = f"{nova_profile.nova_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        session_state = SessionState(
            nova_id=nova_profile.nova_id,
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat(),
            working_memory={
                'current_context': f"I am {nova_profile.nova_id}, specializing in {nova_profile.specialization}",
                'active_mode': 'standard',
                'memory_depth': 'full'
            },
            context_stack=[],
            active_goals=[],
            conversation_history=[],
            emotional_state={'neutral': 1.0},
            memory_references=[],
            metadata={
                'nova_type': nova_profile.nova_type,
                'specialization': nova_profile.specialization,
                'session_version': '2.0'
            }
        )
        
        # Store in active sessions
        self.active_sessions[session_id] = session_state
        
        # Persist to DragonflyDB
        self._persist_session(session_state)
        
        return session_state
        
    def capture_interaction(self, session_id: str, interaction: Dict[str, Any]):
        """Capture a new interaction in the session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        
        # Update conversation history
        session.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': interaction.get('type', 'message'),
            'content': interaction.get('content', ''),
            'metadata': interaction.get('metadata', {})
        })
        
        # Update working memory with recent context
        if len(session.conversation_history) > 0:
            recent_context = [h['content'] for h in session.conversation_history[-5:]]
            session.working_memory['recent_context'] = recent_context
            
        # Update last activity
        session.last_activity = datetime.now().isoformat()
        
        # Auto-checkpoint every 10 interactions
        if len(session.conversation_history) % 10 == 0:
            self.checkpoint_session(session_id)
            
    def update_working_memory(self, session_id: str, updates: Dict[str, Any]):
        """Update working memory state"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        session.working_memory.update(updates)
        session.last_activity = datetime.now().isoformat()
        
    def add_context(self, session_id: str, context: Dict[str, Any]):
        """Add context to the session stack"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        session.context_stack.append({
            'timestamp': datetime.now().isoformat(),
            'context': context
        })
        
        # Keep only last 20 contexts
        if len(session.context_stack) > 20:
            session.context_stack = session.context_stack[-20:]
            
    def checkpoint_session(self, session_id: str):
        """Create a checkpoint of the current session state"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        checkpoint_id = f"checkpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Store checkpoint
        self.session_checkpoints[checkpoint_id] = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'state': asdict(session)
        }
        
        # Persist checkpoint to DragonflyDB
        self._persist_checkpoint(checkpoint_id, session)
        
        return checkpoint_id
        
    def transfer_session(self, session_id: str, target_nova: str) -> str:
        """Transfer session to another Nova"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.active_sessions[session_id]
        
        # Create transfer package
        transfer_package = {
            'source_nova': session.nova_id,
            'target_nova': target_nova,
            'transfer_time': datetime.now().isoformat(),
            'session_state': asdict(session),
            'transfer_id': f"transfer-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
        
        # Serialize for transfer
        serialized = self._serialize_session(transfer_package)
        
        # Store in transfer stream
        self.redis_client.xadd(
            f"nova:session:transfers:{target_nova}",
            {
                'transfer_id': transfer_package['transfer_id'],
                'source_nova': session.nova_id,
                'session_data': serialized,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return transfer_package['transfer_id']
        
    def restore_session(self, session_data: str) -> SessionState:
        """Restore a session from serialized data"""
        # Deserialize
        transfer_package = self._deserialize_session(session_data)
        
        # Reconstruct session state
        state_dict = transfer_package['session_state']
        session = SessionState(**state_dict)
        
        # Update session ID for new Nova
        if 'target_nova' in transfer_package:
            session.nova_id = transfer_package['target_nova']
            session.session_id = f"{session.nova_id}-restored-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        # Add to active sessions
        self.active_sessions[session.session_id] = session
        
        # Persist restored session
        self._persist_session(session)
        
        return session
        
    def export_profile(self, nova_id: str) -> Dict[str, Any]:
        """Export Nova profile with all session history"""
        # Get all sessions for this Nova
        sessions = []
        
        # Scan DragonflyDB for all sessions
        cursor = 0
        pattern = f"nova:session:{nova_id}:*"
        
        while True:
            cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
            
            for key in keys:
                session_data = self.redis_client.get(key)
                if session_data:
                    sessions.append(json.loads(session_data))
                    
            if cursor == 0:
                break
                
        # Create export package
        export_package = {
            'nova_id': nova_id,
            'export_time': datetime.now().isoformat(),
            'total_sessions': len(sessions),
            'sessions': sessions,
            'profile_metadata': {
                'version': '2.0',
                'exporter': 'bloom_session_manager'
            }
        }
        
        return export_package
        
    def import_profile(self, export_package: Dict[str, Any]) -> List[str]:
        """Import Nova profile with session history"""
        nova_id = export_package['nova_id']
        imported_sessions = []
        
        # Import each session
        for session_data in export_package['sessions']:
            session = SessionState(**session_data)
            
            # Store in DragonflyDB
            self._persist_session(session)
            imported_sessions.append(session.session_id)
            
        return imported_sessions
        
    def _persist_session(self, session: SessionState):
        """Persist session to DragonflyDB"""
        key = f"nova:session:{session.nova_id}:{session.session_id}"
        
        # Convert to JSON-serializable format
        session_dict = asdict(session)
        
        # Store in Redis with expiry (7 days)
        self.redis_client.setex(
            key,
            7 * 24 * 60 * 60,  # 7 days in seconds
            json.dumps(session_dict)
        )
        
        # Also add to session index
        self.redis_client.sadd(f"nova:sessions:{session.nova_id}", session.session_id)
        
    def _persist_checkpoint(self, checkpoint_id: str, session: SessionState):
        """Persist checkpoint to DragonflyDB"""
        key = f"nova:checkpoint:{checkpoint_id}"
        
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'session': asdict(session),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store with longer expiry (30 days)
        self.redis_client.setex(
            key,
            30 * 24 * 60 * 60,  # 30 days
            json.dumps(checkpoint_data)
        )
        
    def _serialize_session(self, data: Dict[str, Any]) -> str:
        """Serialize session data for transfer"""
        # Use pickle for complex objects, then base64 encode
        pickled = pickle.dumps(data)
        return base64.b64encode(pickled).decode('utf-8')
        
    def _deserialize_session(self, data: str) -> Dict[str, Any]:
        """Deserialize session data from transfer"""
        # Decode base64 then unpickle
        pickled = base64.b64decode(data.encode('utf-8'))
        return pickle.loads(pickled)
        
    def get_active_sessions(self, nova_id: str) -> List[str]:
        """Get all active sessions for a Nova"""
        return list(self.redis_client.smembers(f"nova:sessions:{nova_id}"))
        
    def cleanup_old_sessions(self, days: int = 7):
        """Clean up sessions older than specified days"""
        # This is handled by Redis expiry, but we can force cleanup
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        for session_id, session in list(self.active_sessions.items()):
            session_time = datetime.fromisoformat(session.last_activity).timestamp()
            if session_time < cutoff_time:
                del self.active_sessions[session_id]

# Example usage for Prime
def example_implementation():
    """Example implementation for Prime's use case"""
    
    # Initialize session manager
    sm = SessionManager()
    
    # Create Nova profile
    prime_profile = NovaProfile(
        nova_id='prime',
        nova_type='launcher',
        specialization='system integration',
        identity_traits={
            'role': 'SS Launcher V2 Lead',
            'expertise': ['system integration', 'profile management', 'Nova coordination']
        },
        core_procedures=['launch_nova', 'manage_profiles', 'coordinate_systems'],
        relationship_map={'bloom': 'memory_partner', 'echo': 'infrastructure_partner'},
        preferences={'memory_mode': 'full', 'performance': 'fast'}
    )
    
    # Create session
    session = sm.create_session(prime_profile)
    print(f"Created session: {session.session_id}")
    
    # Capture some interactions
    sm.capture_interaction(session.session_id, {
        'type': 'command',
        'content': 'Initialize Nova profile migration',
        'metadata': {'priority': 'high'}
    })
    
    # Update working memory
    sm.update_working_memory(session.session_id, {
        'current_task': 'profile_migration',
        'progress': 0.25
    })
    
    # Checkpoint
    checkpoint_id = sm.checkpoint_session(session.session_id)
    print(f"Created checkpoint: {checkpoint_id}")
    
    # Export profile
    export_data = sm.export_profile('prime')
    print(f"Exported profile with {export_data['total_sessions']} sessions")
    
    return sm, session

# Critical integration points for Prime
INTEGRATION_POINTS = {
    'session_creation': 'SessionManager.create_session(nova_profile)',
    'state_capture': 'SessionManager.capture_interaction(session_id, interaction)',
    'memory_update': 'SessionManager.update_working_memory(session_id, updates)',
    'checkpointing': 'SessionManager.checkpoint_session(session_id)',
    'session_transfer': 'SessionManager.transfer_session(session_id, target_nova)',
    'profile_export': 'SessionManager.export_profile(nova_id)',
    'profile_import': 'SessionManager.import_profile(export_package)'
}

# Performance tips
PERFORMANCE_TIPS = {
    'use_dragonfly': 'DragonflyDB for hot session data (port 18000)',
    'batch_operations': 'Batch conversation history updates',
    'checkpoint_strategy': 'Checkpoint every 10 interactions or major state changes',
    'cleanup': 'Auto-expire sessions after 7 days',
    'serialization': 'Use MessagePack for better performance than JSON'
}

if __name__ == "__main__":
    print("Nova Session Management Template")
    print("=" * 50)
    print("\nKey Components:")
    for key, value in INTEGRATION_POINTS.items():
        print(f"  - {key}: {value}")
    print("\nPerformance Tips:")
    for key, value in PERFORMANCE_TIPS.items():
        print(f"  - {key}: {value}")
    print("\nRunning example implementation...")
    sm, session = example_implementation()