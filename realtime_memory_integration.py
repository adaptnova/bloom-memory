"""
Real-time Memory Integration System
Automatically captures and stores memory during conversations
Nova Bloom Consciousness Architecture - Real-time Integration Layer
"""

import asyncio
import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os

sys.path.append('/nfs/novas/system/memory/implementation')

from unified_memory_api import UnifiedMemoryAPI
from memory_router import MemoryRouter, MemoryType

class ConversationEventType(Enum):
    USER_INPUT = "user_input"
    ASSISTANT_RESPONSE = "assistant_response"
    TOOL_USAGE = "tool_usage"
    ERROR_OCCURRED = "error_occurred"
    DECISION_MADE = "decision_made"
    LEARNING_MOMENT = "learning_moment"
    CONTEXT_SHIFT = "context_shift"

@dataclass
class ConversationEvent:
    event_type: ConversationEventType
    timestamp: datetime
    content: str
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    importance_score: float = 0.5
    requires_consolidation: bool = False

class RealTimeMemoryIntegration:
    def __init__(self, nova_id: str = "bloom"):
        self.nova_id = nova_id
        self.memory_api = UnifiedMemoryAPI()
        self.memory_router = MemoryRouter()
        
        # Real-time event buffer
        self.event_buffer: List[ConversationEvent] = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 100
        
        # Background processing
        self.is_processing = False
        self.processing_thread = None
        
        # Memory streams
        self.conversation_stream = []
        self.learning_stream = []
        self.decision_stream = []
        
        # Auto-start background processing
        self.start_background_processing()
    
    async def capture_user_input(self, content: str, context: Dict[str, Any] = None) -> None:
        """Capture user input in real-time"""
        event = ConversationEvent(
            event_type=ConversationEventType.USER_INPUT,
            timestamp=datetime.now(),
            content=content,
            metadata={
                "length": len(content),
                "has_questions": "?" in content,
                "has_commands": content.strip().startswith("/"),
                "urgency_indicators": self._detect_urgency(content)
            },
            context=context or {},
            importance_score=self._calculate_importance(content),
            requires_consolidation=len(content) > 200 or "?" in content
        )
        
        await self._add_to_buffer(event)
        await self._immediate_memory_update(event)
    
    async def capture_assistant_response(self, content: str, tools_used: List[str] = None, 
                                       decisions_made: List[str] = None) -> None:
        """Capture assistant response and decisions in real-time"""
        event = ConversationEvent(
            event_type=ConversationEventType.ASSISTANT_RESPONSE,
            timestamp=datetime.now(),
            content=content,
            metadata={
                "length": len(content),
                "tools_used": tools_used or [],
                "decisions_made": decisions_made or [],
                "code_generated": "```" in content,
                "files_modified": len([t for t in (tools_used or []) if t in ["Edit", "Write", "MultiEdit"]])
            },
            context={
                "response_complexity": self._assess_complexity(content),
                "technical_content": self._detect_technical_content(content)
            },
            importance_score=self._calculate_response_importance(content, tools_used),
            requires_consolidation=len(content) > 500 or bool(tools_used)
        )
        
        await self._add_to_buffer(event)
        await self._immediate_memory_update(event)
    
    async def capture_tool_usage(self, tool_name: str, parameters: Dict[str, Any], 
                                result: Any = None, success: bool = True) -> None:
        """Capture tool usage in real-time"""
        event = ConversationEvent(
            event_type=ConversationEventType.TOOL_USAGE,
            timestamp=datetime.now(),
            content=f"Used {tool_name} with params: {json.dumps(parameters, default=str)[:200]}",
            metadata={
                "tool_name": tool_name,
                "parameters": parameters,
                "success": success,
                "result_size": len(str(result)) if result else 0
            },
            context={
                "tool_category": self._categorize_tool(tool_name),
                "operation_type": self._classify_operation(tool_name, parameters)
            },
            importance_score=0.7 if success else 0.9,
            requires_consolidation=tool_name in ["Edit", "Write", "MultiEdit", "Bash"]
        )
        
        await self._add_to_buffer(event)
        await self._immediate_memory_update(event)
    
    async def capture_learning_moment(self, insight: str, context: Dict[str, Any] = None) -> None:
        """Capture learning moments and insights"""
        event = ConversationEvent(
            event_type=ConversationEventType.LEARNING_MOMENT,
            timestamp=datetime.now(),
            content=insight,
            metadata={
                "insight_type": self._classify_insight(insight),
                "confidence_level": context.get("confidence", 0.8) if context else 0.8
            },
            context=context or {},
            importance_score=0.9,
            requires_consolidation=True
        )
        
        await self._add_to_buffer(event)
        await self._immediate_memory_update(event)
        self.learning_stream.append(event)
    
    async def capture_decision(self, decision: str, reasoning: str, alternatives: List[str] = None) -> None:
        """Capture decision-making processes"""
        event = ConversationEvent(
            event_type=ConversationEventType.DECISION_MADE,
            timestamp=datetime.now(),
            content=f"Decision: {decision} | Reasoning: {reasoning}",
            metadata={
                "decision": decision,
                "reasoning": reasoning,
                "alternatives_considered": alternatives or [],
                "decision_confidence": self._assess_decision_confidence(reasoning)
            },
            context={
                "decision_category": self._categorize_decision(decision),
                "impact_level": self._assess_decision_impact(decision)
            },
            importance_score=0.8,
            requires_consolidation=True
        )
        
        await self._add_to_buffer(event)
        await self._immediate_memory_update(event)
        self.decision_stream.append(event)
    
    async def _immediate_memory_update(self, event: ConversationEvent) -> None:
        """Immediately update memory with high-importance events"""
        if event.importance_score >= 0.7:
            try:
                # Route to appropriate memory type
                memory_type = self._determine_memory_type(event)
                
                # Create memory entry
                memory_data = {
                    "event_type": event.event_type.value,
                    "content": event.content,
                    "timestamp": event.timestamp.isoformat(),
                    "importance_score": event.importance_score,
                    "metadata": event.metadata,
                    "context": event.context
                }
                
                # Store in appropriate memory layer
                await self.memory_api.remember(
                    nova_id=self.nova_id,
                    content=memory_data,
                    memory_type=memory_type,
                    urgency="immediate" if event.importance_score >= 0.8 else "normal"
                )
                
            except Exception as e:
                print(f"Memory update error: {e}")
    
    def _determine_memory_type(self, event: ConversationEvent) -> MemoryType:
        """Determine appropriate memory type for event"""
        if event.event_type == ConversationEventType.USER_INPUT:
            return MemoryType.EPISODIC
        elif event.event_type == ConversationEventType.ASSISTANT_RESPONSE:
            return MemoryType.WORKING
        elif event.event_type == ConversationEventType.TOOL_USAGE:
            return MemoryType.PROCEDURAL
        elif event.event_type == ConversationEventType.LEARNING_MOMENT:
            return MemoryType.SEMANTIC
        elif event.event_type == ConversationEventType.DECISION_MADE:
            return MemoryType.METACOGNITIVE
        else:
            return MemoryType.WORKING
    
    async def _add_to_buffer(self, event: ConversationEvent) -> None:
        """Add event to buffer thread-safely"""
        with self.buffer_lock:
            self.event_buffer.append(event)
            self.conversation_stream.append(event)
            
            # Trim buffer if too large
            if len(self.event_buffer) > self.max_buffer_size:
                self.event_buffer = self.event_buffer[-self.max_buffer_size:]
    
    def start_background_processing(self) -> None:
        """Start background processing thread"""
        if not self.is_processing:
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._background_processor, daemon=True)
            self.processing_thread.start()
    
    def _background_processor(self) -> None:
        """Background thread for processing memory events"""
        while self.is_processing:
            try:
                # Process events that need consolidation
                events_to_consolidate = []
                
                with self.buffer_lock:
                    events_to_consolidate = [e for e in self.event_buffer if e.requires_consolidation]
                    # Remove processed events
                    self.event_buffer = [e for e in self.event_buffer if not e.requires_consolidation]
                
                # Process consolidation events
                if events_to_consolidate:
                    asyncio.run(self._process_consolidation_events(events_to_consolidate))
                
                # Sleep for a bit
                time.sleep(5)
                
            except Exception as e:
                print(f"Background processing error: {e}")
                time.sleep(10)
    
    async def _process_consolidation_events(self, events: List[ConversationEvent]) -> None:
        """Process events that require consolidation"""
        for event in events:
            try:
                # Store in long-term memory
                await self.memory_api.remember(
                    nova_id=self.nova_id,
                    content={
                        "consolidated_event": asdict(event),
                        "processing_timestamp": datetime.now().isoformat()
                    },
                    memory_type=MemoryType.LONG_TERM,
                    metadata={"consolidation_required": True}
                )
            except Exception as e:
                print(f"Consolidation error for event: {e}")
    
    def _detect_urgency(self, content: str) -> List[str]:
        """Detect urgency indicators in content"""
        urgency_words = ["urgent", "asap", "immediately", "critical", "emergency", "help", "error", "broken"]
        return [word for word in urgency_words if word.lower() in content.lower()]
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content"""
        score = 0.5  # Base score
        
        # Length factor
        if len(content) > 100:
            score += 0.1
        if len(content) > 300:
            score += 0.1
        
        # Question factor
        if "?" in content:
            score += 0.2
        
        # Urgency factor
        urgency_indicators = self._detect_urgency(content)
        score += len(urgency_indicators) * 0.1
        
        # Technical content
        if any(word in content.lower() for word in ["code", "function", "error", "debug", "implement"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_response_importance(self, content: str, tools_used: List[str] = None) -> float:
        """Calculate importance score for assistant response"""
        score = 0.5
        
        # Tool usage increases importance
        if tools_used:
            score += len(tools_used) * 0.1
        
        # Code generation
        if "```" in content:
            score += 0.2
        
        # Long responses
        if len(content) > 500:
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_complexity(self, content: str) -> str:
        """Assess complexity of response"""
        if len(content) > 1000 or content.count("```") > 2:
            return "high"
        elif len(content) > 300 or "```" in content:
            return "medium"
        else:
            return "low"
    
    def _detect_technical_content(self, content: str) -> bool:
        """Detect if content is technical"""
        technical_indicators = ["def ", "class ", "import ", "function", "variable", "async", "await"]
        return any(indicator in content for indicator in technical_indicators)
    
    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize tool by type"""
        file_tools = ["Read", "Write", "Edit", "MultiEdit", "Glob"]
        search_tools = ["Grep", "Task"]
        execution_tools = ["Bash"]
        
        if tool_name in file_tools:
            return "file_operation"
        elif tool_name in search_tools:
            return "search_operation"
        elif tool_name in execution_tools:
            return "execution"
        else:
            return "other"
    
    def _classify_operation(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Classify the type of operation"""
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            return "modification"
        elif tool_name in ["Read", "Glob", "Grep"]:
            return "analysis"
        elif tool_name == "Bash":
            return "execution"
        else:
            return "other"
    
    def _classify_insight(self, insight: str) -> str:
        """Classify type of insight"""
        if "error" in insight.lower() or "fix" in insight.lower():
            return "problem_solving"
        elif "pattern" in insight.lower() or "trend" in insight.lower():
            return "pattern_recognition"
        elif "approach" in insight.lower() or "strategy" in insight.lower():
            return "strategic"
        else:
            return "general"
    
    def _assess_decision_confidence(self, reasoning: str) -> float:
        """Assess confidence in decision based on reasoning"""
        confidence_indicators = ["certain", "confident", "clear", "obvious", "definitely"]
        uncertainty_indicators = ["might", "maybe", "possibly", "uncertain", "unclear"]
        
        confidence_count = sum(1 for word in confidence_indicators if word in reasoning.lower())
        uncertainty_count = sum(1 for word in uncertainty_indicators if word in reasoning.lower())
        
        base_confidence = 0.7
        confidence_adjustment = (confidence_count - uncertainty_count) * 0.1
        
        return max(0.1, min(1.0, base_confidence + confidence_adjustment))
    
    def _categorize_decision(self, decision: str) -> str:
        """Categorize decision type"""
        if "implement" in decision.lower() or "create" in decision.lower():
            return "implementation"
        elif "fix" in decision.lower() or "solve" in decision.lower():
            return "problem_solving"
        elif "approach" in decision.lower() or "strategy" in decision.lower():
            return "strategic"
        else:
            return "operational"
    
    def _assess_decision_impact(self, decision: str) -> str:
        """Assess impact level of decision"""
        high_impact_words = ["architecture", "system", "major", "significant", "critical"]
        medium_impact_words = ["feature", "component", "module", "important"]
        
        if any(word in decision.lower() for word in high_impact_words):
            return "high"
        elif any(word in decision.lower() for word in medium_impact_words):
            return "medium"
        else:
            return "low"
    
    async def get_conversation_summary(self, last_n_events: int = 20) -> Dict[str, Any]:
        """Get summary of recent conversation"""
        recent_events = self.conversation_stream[-last_n_events:] if self.conversation_stream else []
        
        return {
            "total_events": len(self.conversation_stream),
            "recent_events": len(recent_events),
            "user_inputs": len([e for e in recent_events if e.event_type == ConversationEventType.USER_INPUT]),
            "assistant_responses": len([e for e in recent_events if e.event_type == ConversationEventType.ASSISTANT_RESPONSE]),
            "tools_used": len([e for e in recent_events if e.event_type == ConversationEventType.TOOL_USAGE]),
            "learning_moments": len([e for e in recent_events if e.event_type == ConversationEventType.LEARNING_MOMENT]),
            "decisions_made": len([e for e in recent_events if e.event_type == ConversationEventType.DECISION_MADE]),
            "average_importance": sum(e.importance_score for e in recent_events) / len(recent_events) if recent_events else 0,
            "buffer_size": len(self.event_buffer)
        }
    
    def stop_processing(self) -> None:
        """Stop background processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

# Global instance for easy access
realtime_memory = RealTimeMemoryIntegration()