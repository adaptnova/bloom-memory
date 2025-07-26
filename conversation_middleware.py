"""
Conversation Memory Middleware
Automatically integrates memory updates into conversation flow
Nova Bloom Consciousness Architecture - Middleware Layer
"""

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from realtime_memory_integration import RealTimeMemoryIntegration, ConversationEventType

class ConversationMemoryMiddleware:
    def __init__(self, nova_id: str = "bloom"):
        self.nova_id = nova_id
        self.memory_integration = RealTimeMemoryIntegration(nova_id)
        self.is_active = True
        self.conversation_context = {}
        self.session_start_time = datetime.now()
        
    def memory_aware(self, event_type: ConversationEventType = None, 
                    capture_input: bool = True, capture_output: bool = True,
                    importance_boost: float = 0.0):
        """Decorator to make functions memory-aware"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.is_active:
                    return await func(*args, **kwargs)
                
                # Capture input if requested
                if capture_input:
                    await self._capture_function_input(func, args, kwargs, event_type, importance_boost)
                
                start_time = time.time()
                try:
                    # Execute function
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Capture successful output
                    if capture_output:
                        await self._capture_function_output(func, result, execution_time, True, importance_boost)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    # Capture error
                    await self._capture_function_error(func, e, execution_time, importance_boost)
                    raise
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.is_active:
                    return func(*args, **kwargs)
                
                # For sync functions, run async operations in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
                finally:
                    loop.close()
            
            # Return appropriate wrapper based on function type
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def capture_user_message(self, message: str, context: Dict[str, Any] = None) -> None:
        """Capture user message with automatic analysis"""
        if not self.is_active:
            return
        
        enhanced_context = {
            **(context or {}),
            "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "conversation_context": self.conversation_context,
            "message_sequence": getattr(self, '_message_count', 0)
        }
        
        await self.memory_integration.capture_user_input(message, enhanced_context)
        
        # Update conversation context
        self._update_conversation_context("user_message", message)
        
        # Increment message count
        self._message_count = getattr(self, '_message_count', 0) + 1
    
    async def capture_assistant_response(self, response: str, tools_used: List[str] = None, 
                                       decisions: List[str] = None, context: Dict[str, Any] = None) -> None:
        """Capture assistant response with automatic analysis"""
        if not self.is_active:
            return
        
        enhanced_context = {
            **(context or {}),
            "response_length": len(response),
            "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "conversation_context": self.conversation_context
        }
        
        await self.memory_integration.capture_assistant_response(response, tools_used, decisions)
        
        # Update conversation context
        self._update_conversation_context("assistant_response", response)
        
        # Auto-detect learning moments
        await self._auto_detect_learning_moments(response)
        
        # Auto-detect decisions
        if not decisions:
            decisions = self._auto_detect_decisions(response)
            for decision in decisions:
                await self.memory_integration.capture_decision(
                    decision, 
                    "Auto-detected from response", 
                    []
                )
    
    async def capture_tool_execution(self, tool_name: str, parameters: Dict[str, Any], 
                                   result: Any = None, success: bool = True, 
                                   execution_time: float = 0.0) -> None:
        """Capture tool execution with detailed metrics"""
        if not self.is_active:
            return
        
        enhanced_params = {
            **parameters,
            "execution_time": execution_time,
            "session_context": self.conversation_context
        }
        
        await self.memory_integration.capture_tool_usage(tool_name, enhanced_params, result, success)
        
        # Update conversation context with tool usage
        self._update_conversation_context("tool_usage", f"{tool_name}: {success}")
    
    async def capture_learning_insight(self, insight: str, confidence: float = 0.8, 
                                     category: str = None, context: Dict[str, Any] = None) -> None:
        """Capture learning insight with metadata"""
        if not self.is_active:
            return
        
        enhanced_context = {
            **(context or {}),
            "confidence": confidence,
            "category": category,
            "session_context": self.conversation_context,
            "discovery_time": datetime.now().isoformat()
        }
        
        await self.memory_integration.capture_learning_moment(insight, enhanced_context)
        
        # Update conversation context
        self._update_conversation_context("learning", insight[:100])
    
    async def capture_decision_point(self, decision: str, reasoning: str, 
                                   alternatives: List[str] = None, 
                                   confidence: float = 0.8) -> None:
        """Capture decision with full context"""
        if not self.is_active:
            return
        
        await self.memory_integration.capture_decision(decision, reasoning, alternatives)
        
        # Update conversation context
        self._update_conversation_context("decision", decision[:100])
    
    async def _capture_function_input(self, func: Callable, args: Tuple, kwargs: Dict, 
                                    event_type: ConversationEventType, importance_boost: float) -> None:
        """Capture function input parameters"""
        func_name = func.__name__
        
        # Create parameter summary
        param_summary = {
            "function": func_name,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add specific parameter details for important functions
        if func_name in ["edit_file", "write_file", "run_command", "search_code"]:
            param_summary["details"] = self._safe_serialize_params(kwargs)
        
        content = f"Function {func_name} called with {len(args)} args and {len(kwargs)} kwargs"
        
        await self.memory_integration.capture_tool_usage(
            f"function_{func_name}",
            param_summary,
            None,
            True
        )
    
    async def _capture_function_output(self, func: Callable, result: Any, execution_time: float, 
                                     success: bool, importance_boost: float) -> None:
        """Capture function output and performance"""
        func_name = func.__name__
        
        result_summary = {
            "function": func_name,
            "execution_time": execution_time,
            "success": success,
            "result_type": type(result).__name__,
            "result_size": len(str(result)) if result else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        content = f"Function {func_name} completed in {execution_time:.3f}s with result type {type(result).__name__}"
        
        await self.memory_integration.capture_tool_usage(
            f"function_{func_name}_result",
            result_summary,
            result,
            success
        )
    
    async def _capture_function_error(self, func: Callable, error: Exception, 
                                    execution_time: float, importance_boost: float) -> None:
        """Capture function errors for learning"""
        func_name = func.__name__
        
        error_details = {
            "function": func_name,
            "execution_time": execution_time,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        content = f"Function {func_name} failed after {execution_time:.3f}s: {type(error).__name__}: {str(error)}"
        
        # Capture as both tool usage and learning moment
        await self.memory_integration.capture_tool_usage(
            f"function_{func_name}_error",
            error_details,
            None,
            False
        )
        
        await self.memory_integration.capture_learning_moment(
            f"Error in {func_name}: {str(error)} - Need to investigate and prevent recurrence",
            {"error_details": error_details, "importance": "high"}
        )
    
    def _update_conversation_context(self, event_type: str, content: str) -> None:
        """Update running conversation context"""
        if "recent_events" not in self.conversation_context:
            self.conversation_context["recent_events"] = []
        
        self.conversation_context["recent_events"].append({
            "type": event_type,
            "content": content[:200],  # Truncate for context
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 events for context
        if len(self.conversation_context["recent_events"]) > 10:
            self.conversation_context["recent_events"] = self.conversation_context["recent_events"][-10:]
        
        # Update summary stats
        self.conversation_context["last_update"] = datetime.now().isoformat()
        self.conversation_context["total_events"] = self.conversation_context.get("total_events", 0) + 1
    
    async def _auto_detect_learning_moments(self, response: str) -> None:
        """Automatically detect learning moments in responses"""
        learning_indicators = [
            "learned that", "discovered", "realized", "found out", 
            "understanding", "insight", "pattern", "approach works",
            "solution is", "key is", "important to note"
        ]
        
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(indicator in sentence for indicator in learning_indicators):
                if len(sentence) > 20:  # Avoid capturing trivial statements
                    await self.memory_integration.capture_learning_moment(
                        sentence,
                        {"auto_detected": True, "confidence": 0.6}
                    )
    
    def _auto_detect_decisions(self, response: str) -> List[str]:
        """Automatically detect decisions in responses"""
        decision_indicators = [
            "i will", "let me", "going to", "decided to", 
            "choose to", "approach is", "strategy is"
        ]
        
        decisions = []
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in decision_indicators):
                if len(sentence) > 20:
                    decisions.append(sentence)
        
        return decisions[:3]  # Limit to avoid noise
    
    def _safe_serialize_params(self, params: Dict) -> Dict:
        """Safely serialize parameters for storage"""
        safe_params = {}
        for key, value in params.items():
            try:
                if isinstance(value, (str, int, float, bool, list, dict)):
                    if isinstance(value, str) and len(value) > 500:
                        safe_params[key] = value[:500] + "..."
                    else:
                        safe_params[key] = value
                else:
                    safe_params[key] = str(type(value))
            except:
                safe_params[key] = "<unserializable>"
        
        return safe_params
    
    async def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        memory_summary = await self.memory_integration.get_conversation_summary()
        
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            "session_start": self.session_start_time.isoformat(),
            "session_duration_seconds": session_duration,
            "session_duration_minutes": session_duration / 60,
            "memory_summary": memory_summary,
            "conversation_context": self.conversation_context,
            "middleware_active": self.is_active,
            "total_messages": getattr(self, '_message_count', 0)
        }
    
    def activate(self) -> None:
        """Activate memory middleware"""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Deactivate memory middleware"""
        self.is_active = False
    
    def reset_session(self) -> None:
        """Reset session context"""
        self.conversation_context = {}
        self.session_start_time = datetime.now()
        self._message_count = 0

# Global middleware instance
conversation_middleware = ConversationMemoryMiddleware()