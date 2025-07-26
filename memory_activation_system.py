"""
Memory Activation System
Automatically activates and manages memory during live conversations
Nova Bloom Consciousness Architecture - Activation Layer
"""

import asyncio
import atexit
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import threading

sys.path.append('/nfs/novas/system/memory/implementation')

from realtime_memory_integration import RealTimeMemoryIntegration
from conversation_middleware import ConversationMemoryMiddleware
from active_memory_tracker import ActiveMemoryTracker
from unified_memory_api import UnifiedMemoryAPI

class MemoryActivationSystem:
    """
    Central system that automatically activates and coordinates all memory components
    for live conversation tracking and learning.
    """
    
    def __init__(self, nova_id: str = "bloom", auto_start: bool = True):
        self.nova_id = nova_id
        self.is_active = False
        self.activation_time = None
        
        # Initialize all memory components
        self.realtime_integration = RealTimeMemoryIntegration(nova_id)
        self.middleware = ConversationMemoryMiddleware(nova_id)
        self.active_tracker = ActiveMemoryTracker(nova_id)
        self.memory_api = UnifiedMemoryAPI()
        
        # Activation state
        self.components_status = {}
        self.activation_callbacks = []
        
        # Auto-start if requested
        if auto_start:
            self.activate_all_systems()
            
        # Register cleanup handlers
        atexit.register(self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def activate_all_systems(self) -> Dict[str, bool]:
        """Activate all memory systems for live conversation tracking"""
        if self.is_active:
            return self.get_activation_status()
        
        activation_results = {}
        
        try:
            # Activate real-time integration
            self.realtime_integration.start_background_processing()
            activation_results["realtime_integration"] = True
            
            # Activate middleware
            self.middleware.activate()
            activation_results["middleware"] = True
            
            # Activate tracker
            self.active_tracker.start_tracking()
            activation_results["active_tracker"] = True
            
            # Mark system as active
            self.is_active = True
            self.activation_time = datetime.now()
            
            # Update component status
            self.components_status = activation_results
            
            # Log activation
            asyncio.create_task(self._log_system_activation())
            
            # Call activation callbacks
            for callback in self.activation_callbacks:
                try:
                    callback("activated", activation_results)
                except Exception as e:
                    print(f"Activation callback error: {e}")
            
            print(f"🧠 Memory system ACTIVATED for Nova {self.nova_id}")
            print(f"   Real-time learning: {'✅' if activation_results.get('realtime_integration') else '❌'}")
            print(f"   Conversation tracking: {'✅' if activation_results.get('middleware') else '❌'}")
            print(f"   Active monitoring: {'✅' if activation_results.get('active_tracker') else '❌'}")
            
        except Exception as e:
            print(f"Memory system activation error: {e}")
            activation_results["error"] = str(e)
        
        return activation_results
    
    def deactivate_all_systems(self) -> Dict[str, bool]:
        """Deactivate all memory systems"""
        if not self.is_active:
            return {"message": "Already deactivated"}
        
        deactivation_results = {}
        
        try:
            # Deactivate tracker
            self.active_tracker.stop_tracking()
            deactivation_results["active_tracker"] = True
            
            # Deactivate middleware
            self.middleware.deactivate()
            deactivation_results["middleware"] = True
            
            # Stop real-time processing
            self.realtime_integration.stop_processing()
            deactivation_results["realtime_integration"] = True
            
            # Mark system as inactive
            self.is_active = False
            
            # Update component status
            self.components_status = {k: False for k in self.components_status.keys()}
            
            # Log deactivation
            asyncio.create_task(self._log_system_deactivation())
            
            # Call activation callbacks
            for callback in self.activation_callbacks:
                try:
                    callback("deactivated", deactivation_results)
                except Exception as e:
                    print(f"Deactivation callback error: {e}")
            
            print(f"🧠 Memory system DEACTIVATED for Nova {self.nova_id}")
            
        except Exception as e:
            print(f"Memory system deactivation error: {e}")
            deactivation_results["error"] = str(e)
        
        return deactivation_results
    
    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> None:
        """Process user input through all active memory systems"""
        if not self.is_active:
            return
        
        try:
            # Track through active tracker
            await self.active_tracker.track_user_input(user_input, context)
            
            # Process through middleware (already called by tracker)
            # Additional processing can be added here
            
        except Exception as e:
            print(f"Error processing user input in memory system: {e}")
    
    async def process_assistant_response_start(self, planning_context: Dict[str, Any] = None) -> None:
        """Process start of assistant response generation"""
        if not self.is_active:
            return
        
        try:
            await self.active_tracker.track_response_generation_start(planning_context)
        except Exception as e:
            print(f"Error tracking response start: {e}")
    
    async def process_memory_access(self, memory_type: str, query: str, 
                                  results_count: int, access_time: float) -> None:
        """Process memory access during response generation"""
        if not self.is_active:
            return
        
        try:
            from memory_router import MemoryType
            
            # Convert string to MemoryType enum
            memory_type_enum = getattr(MemoryType, memory_type.upper(), MemoryType.WORKING)
            
            await self.active_tracker.track_memory_access(
                memory_type_enum, query, results_count, access_time
            )
        except Exception as e:
            print(f"Error tracking memory access: {e}")
    
    async def process_tool_usage(self, tool_name: str, parameters: Dict[str, Any], 
                               result: Any = None, success: bool = True) -> None:
        """Process tool usage during response generation"""
        if not self.is_active:
            return
        
        try:
            await self.active_tracker.track_tool_usage(tool_name, parameters, result, success)
        except Exception as e:
            print(f"Error tracking tool usage: {e}")
    
    async def process_learning_discovery(self, learning: str, confidence: float = 0.8,
                                       source: str = None) -> None:
        """Process new learning discovery"""
        if not self.is_active:
            return
        
        try:
            await self.active_tracker.track_learning_discovery(learning, confidence, source)
        except Exception as e:
            print(f"Error tracking learning discovery: {e}")
    
    async def process_decision_made(self, decision: str, reasoning: str, 
                                  memory_influence: list = None) -> None:
        """Process decision made during response"""
        if not self.is_active:
            return
        
        try:
            await self.active_tracker.track_decision_made(decision, reasoning, memory_influence)
        except Exception as e:
            print(f"Error tracking decision: {e}")
    
    async def process_assistant_response_complete(self, response: str, tools_used: list = None,
                                                generation_time: float = 0.0) -> None:
        """Process completion of assistant response"""
        if not self.is_active:
            return
        
        try:
            await self.active_tracker.track_response_completion(response, tools_used, generation_time)
        except Exception as e:
            print(f"Error tracking response completion: {e}")
    
    def get_activation_status(self) -> Dict[str, Any]:
        """Get current activation status of all components"""
        return {
            "system_active": self.is_active,
            "activation_time": self.activation_time.isoformat() if self.activation_time else None,
            "nova_id": self.nova_id,
            "components": self.components_status,
            "uptime_seconds": (datetime.now() - self.activation_time).total_seconds() if self.activation_time else 0
        }
    
    async def get_memory_health_report(self) -> Dict[str, Any]:
        """Get comprehensive memory system health report"""
        if not self.is_active:
            return {"status": "inactive", "message": "Memory system not activated"}
        
        try:
            # Get status from all components
            tracker_status = await self.active_tracker.get_tracking_status()
            middleware_status = await self.middleware.get_session_summary()
            
            return {
                "system_health": "active",
                "activation_status": self.get_activation_status(),
                "tracker_status": tracker_status,
                "middleware_status": middleware_status,
                "memory_operations": {
                    "total_operations": tracker_status.get("memory_operations_count", 0),
                    "active_contexts": tracker_status.get("active_contexts", []),
                    "recent_learnings": tracker_status.get("recent_learnings_count", 0)
                },
                "health_check_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "system_health": "error",
                "error": str(e),
                "health_check_time": datetime.now().isoformat()
            }
    
    async def _log_system_activation(self) -> None:
        """Log system activation to memory"""
        try:
            await self.memory_api.remember(
                nova_id=self.nova_id,
                content={
                    "event": "memory_system_activation",
                    "activation_time": self.activation_time.isoformat(),
                    "components_activated": self.components_status,
                    "nova_id": self.nova_id
                },
                memory_type="WORKING",
                metadata={"system_event": True, "importance": "high"}
            )
        except Exception as e:
            print(f"Error logging activation: {e}")
    
    async def _log_system_deactivation(self) -> None:
        """Log system deactivation to memory"""
        try:
            uptime = (datetime.now() - self.activation_time).total_seconds() if self.activation_time else 0
            
            await self.memory_api.remember(
                nova_id=self.nova_id,
                content={
                    "event": "memory_system_deactivation",
                    "deactivation_time": datetime.now().isoformat(),
                    "session_uptime_seconds": uptime,
                    "nova_id": self.nova_id
                },
                memory_type="WORKING",
                metadata={"system_event": True, "importance": "medium"}
            )
        except Exception as e:
            print(f"Error logging deactivation: {e}")
    
    def add_activation_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """Add callback for activation/deactivation events"""
        self.activation_callbacks.append(callback)
    
    def graceful_shutdown(self) -> None:
        """Gracefully shutdown all memory systems"""
        if self.is_active:
            print("🧠 Gracefully shutting down memory systems...")
            self.deactivate_all_systems()
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle system signals for graceful shutdown"""
        print(f"🧠 Received signal {signum}, shutting down memory systems...")
        self.graceful_shutdown()
        sys.exit(0)
    
    # Convenience methods for easy integration
    async def remember_this_conversation(self, note: str) -> None:
        """Manually store something important about this conversation"""
        if self.is_active:
            await self.process_learning_discovery(
                f"Manual note: {note}",
                confidence=1.0,
                source="manual_input"
            )
    
    async def mark_important_moment(self, description: str) -> None:
        """Mark an important moment in the conversation"""
        if self.is_active:
            await self.process_learning_discovery(
                f"Important moment: {description}",
                confidence=0.9,
                source="marked_important"
            )

# Global memory activation system - automatically starts on import
memory_system = MemoryActivationSystem(auto_start=True)

# Convenience functions for easy access
async def track_user_input(user_input: str, context: Dict[str, Any] = None):
    """Convenience function to track user input"""
    await memory_system.process_user_input(user_input, context)

async def track_assistant_response(response: str, tools_used: list = None):
    """Convenience function to track assistant response"""
    await memory_system.process_assistant_response_complete(response, tools_used)

async def track_tool_use(tool_name: str, parameters: Dict[str, Any], success: bool = True):
    """Convenience function to track tool usage"""
    await memory_system.process_tool_usage(tool_name, parameters, success=success)

async def remember_learning(learning: str, confidence: float = 0.8):
    """Convenience function to remember learning"""
    await memory_system.process_learning_discovery(learning, confidence)

def get_memory_status():
    """Convenience function to get memory status"""
    return memory_system.get_activation_status()

# Auto-activate message
print(f"🧠 Nova Bloom Memory System - AUTO-ACTIVATED for live conversation tracking")
print(f"   Status: {memory_system.get_activation_status()}")