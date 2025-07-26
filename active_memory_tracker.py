"""
Active Memory Tracker
Continuously tracks and updates memory during live conversations
Nova Bloom Consciousness Architecture - Live Tracking System
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from collections import deque
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from realtime_memory_integration import RealTimeMemoryIntegration
from conversation_middleware import ConversationMemoryMiddleware
from unified_memory_api import UnifiedMemoryAPI
from memory_router import MemoryType

@dataclass
class MemorySnapshot:
    timestamp: datetime
    conversation_state: Dict[str, Any]
    active_contexts: List[str]
    recent_learnings: List[str]
    pending_consolidations: int
    memory_health: Dict[str, Any]

class ActiveMemoryTracker:
    def __init__(self, nova_id: str = "bloom"):
        self.nova_id = nova_id
        self.memory_integration = RealTimeMemoryIntegration(nova_id)
        self.middleware = ConversationMemoryMiddleware(nova_id)
        self.memory_api = UnifiedMemoryAPI()
        
        # Tracking state
        self.is_tracking = False
        self.tracking_thread = None
        self.memory_snapshots = deque(maxlen=100)
        
        # Live conversation state
        self.current_conversation_id = self._generate_conversation_id()
        self.conversation_start_time = datetime.now()
        self.active_contexts: Set[str] = set()
        self.recent_learnings: List[Dict[str, Any]] = []
        self.response_being_generated = False
        
        # Memory health monitoring
        self.memory_operations_count = 0
        self.last_consolidation_time = datetime.now()
        self.consolidation_queue_size = 0
        
        # Auto-start tracking
        self.start_tracking()
    
    def start_tracking(self) -> None:
        """Start active memory tracking"""
        if not self.is_tracking:
            self.is_tracking = True
            self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
            self.tracking_thread.start()
            
            # Activate middleware
            self.middleware.activate()
            
            print(f"Active memory tracking started for Nova {self.nova_id}")
    
    def stop_tracking(self) -> None:
        """Stop active memory tracking"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
        
        self.middleware.deactivate()
        print(f"Active memory tracking stopped for Nova {self.nova_id}")
    
    async def track_conversation_start(self, initial_context: str = None) -> None:
        """Track the start of a new conversation"""
        self.current_conversation_id = self._generate_conversation_id()
        self.conversation_start_time = datetime.now()
        self.active_contexts.clear()
        self.recent_learnings.clear()
        
        if initial_context:
            self.active_contexts.add(initial_context)
        
        # Log conversation start
        await self.memory_integration.capture_learning_moment(
            f"Starting new conversation session: {self.current_conversation_id}",
            {
                "conversation_id": self.current_conversation_id,
                "start_time": self.conversation_start_time.isoformat(),
                "initial_context": initial_context
            }
        )
    
    async def track_user_input(self, user_input: str, context: Dict[str, Any] = None) -> None:
        """Track user input and update conversation state"""
        # Capture through middleware
        await self.middleware.capture_user_message(user_input, context)
        
        # Update active contexts
        detected_contexts = self._extract_contexts_from_input(user_input)
        self.active_contexts.update(detected_contexts)
        
        # Analyze input for memory implications
        await self._analyze_input_implications(user_input)
        
        # Update conversation state
        await self._update_conversation_state("user_input", user_input)
    
    async def track_response_generation_start(self, planning_context: Dict[str, Any] = None) -> None:
        """Track when response generation begins"""
        self.response_being_generated = True
        
        await self.memory_integration.capture_learning_moment(
            "Response generation started - accessing memory for context",
            {
                "conversation_id": self.current_conversation_id,
                "active_contexts": list(self.active_contexts),
                "planning_context": planning_context or {}
            }
        )
    
    async def track_memory_access(self, memory_type: MemoryType, query: str, 
                                results_count: int, access_time: float) -> None:
        """Track memory access during response generation"""
        await self.memory_integration.capture_tool_usage(
            "memory_access",
            {
                "memory_type": memory_type.value,
                "query": query[:200],
                "results_count": results_count,
                "access_time": access_time,
                "conversation_id": self.current_conversation_id
            },
            f"Retrieved {results_count} results in {access_time:.3f}s",
            True
        )
        
        self.memory_operations_count += 1
    
    async def track_decision_made(self, decision: str, reasoning: str, 
                                memory_influence: List[str] = None) -> None:
        """Track decisions made during response generation"""
        await self.middleware.capture_decision_point(
            decision, 
            reasoning,
            [],  # alternatives
            0.8   # confidence
        )
        
        # Track memory influence on decision
        if memory_influence:
            await self.memory_integration.capture_learning_moment(
                f"Memory influenced decision: {decision}",
                {
                    "decision": decision,
                    "memory_sources": memory_influence,
                    "conversation_id": self.current_conversation_id
                }
            )
    
    async def track_tool_usage(self, tool_name: str, parameters: Dict[str, Any], 
                             result: Any = None, success: bool = True) -> None:
        """Track tool usage during response generation"""
        execution_time = parameters.get("execution_time", 0.0)
        
        await self.middleware.capture_tool_execution(
            tool_name,
            parameters,
            result,
            success,
            execution_time
        )
        
        # Update active contexts based on tool usage
        if tool_name in ["Read", "Grep", "Glob"] and success:
            if "file_path" in parameters:
                self.active_contexts.add(f"file:{parameters['file_path']}")
            if "pattern" in parameters:
                self.active_contexts.add(f"search:{parameters['pattern']}")
    
    async def track_learning_discovery(self, learning: str, confidence: float = 0.8,
                                     source: str = None) -> None:
        """Track new learning discovered during conversation"""
        learning_entry = {
            "content": learning,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "conversation_id": self.current_conversation_id
        }
        
        self.recent_learnings.append(learning_entry)
        
        # Keep only recent learnings
        if len(self.recent_learnings) > 20:
            self.recent_learnings = self.recent_learnings[-20:]
        
        await self.middleware.capture_learning_insight(learning, confidence, source)
    
    async def track_response_completion(self, response: str, tools_used: List[str] = None,
                                      generation_time: float = 0.0) -> None:
        """Track completion of response generation"""
        self.response_being_generated = False
        
        # Capture response
        await self.middleware.capture_assistant_response(
            response,
            tools_used,
            [],  # decisions auto-detected
            {
                "generation_time": generation_time,
                "conversation_id": self.current_conversation_id,
                "active_contexts_count": len(self.active_contexts)
            }
        )
        
        # Analyze response for new contexts
        new_contexts = self._extract_contexts_from_response(response)
        self.active_contexts.update(new_contexts)
        
        # Update conversation state
        await self._update_conversation_state("assistant_response", response)
        
        # Check if consolidation is needed
        await self._check_consolidation_trigger()
    
    async def _analyze_input_implications(self, user_input: str) -> None:
        """Analyze user input for memory storage implications"""
        # Detect if user is asking about past events
        if any(word in user_input.lower() for word in ["remember", "recall", "what did", "when did", "how did"]):
            await self.memory_integration.capture_learning_moment(
                "User requesting memory recall - may need to access episodic memory",
                {"input_type": "memory_query", "user_input": user_input[:200]}
            )
        
        # Detect if user is providing new information
        if any(phrase in user_input.lower() for phrase in ["let me tell you", "by the way", "also", "additionally"]):
            await self.memory_integration.capture_learning_moment(
                "User providing new information - store in episodic memory",
                {"input_type": "information_provided", "user_input": user_input[:200]}
            )
        
        # Detect task/goal changes
        if any(word in user_input.lower() for word in ["now", "instead", "change", "different", "new task"]):
            await self.memory_integration.capture_learning_moment(
                "Potential task/goal change detected",
                {"input_type": "context_shift", "user_input": user_input[:200]}
            )
    
    def _extract_contexts_from_input(self, user_input: str) -> Set[str]:
        """Extract context indicators from user input"""
        contexts = set()
        
        # File/path contexts
        if "/" in user_input and ("file" in user_input.lower() or "path" in user_input.lower()):
            contexts.add("file_operations")
        
        # Code contexts
        if any(word in user_input.lower() for word in ["code", "function", "class", "implement", "debug"]):
            contexts.add("coding")
        
        # System contexts
        if any(word in user_input.lower() for word in ["server", "database", "system", "architecture"]):
            contexts.add("system_architecture")
        
        # Memory contexts
        if any(word in user_input.lower() for word in ["memory", "remember", "store", "recall"]):
            contexts.add("memory_management")
        
        return contexts
    
    def _extract_contexts_from_response(self, response: str) -> Set[str]:
        """Extract context indicators from assistant response"""
        contexts = set()
        
        # Tool usage contexts
        if "```" in response:
            contexts.add("code_generation")
        
        # File operation contexts
        if any(tool in response for tool in ["Read", "Write", "Edit", "Glob", "Grep"]):
            contexts.add("file_operations")
        
        # Decision contexts
        if any(phrase in response.lower() for phrase in ["i will", "let me", "going to", "approach"]):
            contexts.add("decision_making")
        
        return contexts
    
    async def _update_conversation_state(self, event_type: str, content: str) -> None:
        """Update the current conversation state"""
        state_update = {
            "event_type": event_type,
            "content_length": len(content),
            "timestamp": datetime.now().isoformat(),
            "active_contexts": list(self.active_contexts),
            "conversation_id": self.current_conversation_id
        }
        
        # Store state update in working memory
        await self.memory_api.remember(
            nova_id=self.nova_id,
            content=state_update,
            memory_type=MemoryType.WORKING,
            metadata={"conversation_state": True}
        )
    
    async def _check_consolidation_trigger(self) -> None:
        """Check if memory consolidation should be triggered"""
        time_since_last_consolidation = datetime.now() - self.last_consolidation_time
        
        # Trigger consolidation if:
        # 1. More than 50 memory operations since last consolidation
        # 2. More than 10 minutes since last consolidation
        # 3. Conversation context is getting large
        
        should_consolidate = (
            self.memory_operations_count > 50 or
            time_since_last_consolidation > timedelta(minutes=10) or
            len(self.active_contexts) > 15
        )
        
        if should_consolidate:
            await self._trigger_consolidation()
    
    async def _trigger_consolidation(self) -> None:
        """Trigger memory consolidation process"""
        await self.memory_integration.capture_learning_moment(
            "Triggering memory consolidation - processing recent conversation events",
            {
                "consolidation_trigger": "automatic",
                "memory_operations_count": self.memory_operations_count,
                "active_contexts_count": len(self.active_contexts),
                "conversation_id": self.current_conversation_id
            }
        )
        
        # Reset counters
        self.memory_operations_count = 0
        self.last_consolidation_time = datetime.now()
        
        # Create consolidation task (would be processed by consolidation engine)
        consolidation_data = {
            "conversation_id": self.current_conversation_id,
            "consolidation_timestamp": datetime.now().isoformat(),
            "contexts_to_consolidate": list(self.active_contexts),
            "recent_learnings": self.recent_learnings
        }
        
        await self.memory_api.remember(
            nova_id=self.nova_id,
            content=consolidation_data,
            memory_type=MemoryType.LONG_TERM,
            metadata={"consolidation_task": True}
        )
    
    def _tracking_loop(self) -> None:
        """Main tracking loop running in background thread"""
        while self.is_tracking:
            try:
                # Create memory snapshot
                snapshot = MemorySnapshot(
                    timestamp=datetime.now(),
                    conversation_state={
                        "conversation_id": self.current_conversation_id,
                        "active_contexts": list(self.active_contexts),
                        "response_being_generated": self.response_being_generated,
                        "session_duration": (datetime.now() - self.conversation_start_time).total_seconds()
                    },
                    active_contexts=list(self.active_contexts),
                    recent_learnings=[l["content"] for l in self.recent_learnings[-5:]],
                    pending_consolidations=self.consolidation_queue_size,
                    memory_health={
                        "operations_count": self.memory_operations_count,
                        "last_consolidation": self.last_consolidation_time.isoformat(),
                        "tracking_active": self.is_tracking
                    }
                )
                
                self.memory_snapshots.append(snapshot)
                
                # Sleep for tracking interval
                time.sleep(30)  # Take snapshot every 30 seconds
                
            except Exception as e:
                print(f"Memory tracking error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _generate_conversation_id(self) -> str:
        """Generate unique conversation ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"conv_{self.nova_id}_{timestamp}"
    
    async def get_tracking_status(self) -> Dict[str, Any]:
        """Get current tracking status"""
        return {
            "tracking_active": self.is_tracking,
            "conversation_id": self.current_conversation_id,
            "session_duration": (datetime.now() - self.conversation_start_time).total_seconds(),
            "active_contexts": list(self.active_contexts),
            "recent_learnings_count": len(self.recent_learnings),
            "memory_operations_count": self.memory_operations_count,
            "response_being_generated": self.response_being_generated,
            "snapshots_count": len(self.memory_snapshots),
            "last_consolidation": self.last_consolidation_time.isoformat()
        }
    
    async def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        session_summary = await self.middleware.get_session_summary()
        tracking_status = await self.get_tracking_status()
        
        return {
            "conversation_overview": {
                "id": self.current_conversation_id,
                "duration_minutes": tracking_status["session_duration"] / 60,
                "contexts_explored": len(self.active_contexts),
                "learnings_discovered": len(self.recent_learnings)
            },
            "memory_activity": {
                "operations_performed": self.memory_operations_count,
                "last_consolidation": self.last_consolidation_time.isoformat(),
                "consolidations_needed": self.consolidation_queue_size
            },
            "session_details": session_summary,
            "tracking_details": tracking_status
        }

# Global tracker instance
active_memory_tracker = ActiveMemoryTracker()