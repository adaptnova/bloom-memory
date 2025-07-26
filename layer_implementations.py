#!/usr/bin/env python3
"""
Nova Memory System - Specific Layer Implementations (1-10)
Implements the first 10 layers for immediate and short-term processing
"""

import json
import asyncio
from datetime import timedelta
from typing import Dict, List, Any, Optional

from memory_layers import (
    MemoryLayer, DragonflyMemoryLayer, MemoryScope, 
    MemoryImportance, MemoryEntry
)

# Layer 1: Sensory Buffer
class SensoryBufferLayer(DragonflyMemoryLayer):
    """
    Layer 1: Raw sensory input stream (0.5-30 seconds)
    Ultra-low latency, minimal processing
    """
    
    def __init__(self):
        super().__init__(
            layer_id=1,
            layer_name="sensory_buffer",
            capacity=1000,  # Rolling buffer of 1000 entries
            retention=timedelta(seconds=30),
            scope=MemoryScope.VOLATILE
        )
        self.buffer_ttl = 30  # seconds
        
    async def write(self, nova_id: str, data: Dict[str, Any], **kwargs) -> str:
        """Write with automatic TTL"""
        memory_id = await super().write(nova_id, data, **kwargs)
        
        # Set TTL on the entry
        if self.connection:
            stream_key = self.stream_key_template.format(
                nova_id=nova_id,
                layer_name=self.layer_name
            )
            self.connection.expire(f"{stream_key}:lookup:{memory_id}", self.buffer_ttl)
            
        return memory_id

# Layer 2: Attention Filter
class AttentionFilterLayer(DragonflyMemoryLayer):
    """
    Layer 2: Filtered attention stream (1-60 seconds)
    Filters sensory input based on importance and relevance
    """
    
    def __init__(self):
        super().__init__(
            layer_id=2,
            layer_name="attention_filter",
            capacity=500,
            retention=timedelta(seconds=60),
            scope=MemoryScope.VOLATILE
        )
        self.importance_threshold = 0.3
        
    async def write(self, nova_id: str, data: Dict[str, Any], 
                   importance: float = 0.5, **kwargs) -> str:
        """Only write if importance exceeds threshold"""
        if importance < self.importance_threshold:
            return ""  # Filtered out
            
        # Enhance data with attention metadata
        data['attention_score'] = importance
        data['attention_timestamp'] = self.stats['last_operation']['timestamp']
        
        return await super().write(nova_id, data, importance=importance, **kwargs)

# Layer 3: Working Memory
class WorkingMemoryLayer(DragonflyMemoryLayer):
    """
    Layer 3: Active manipulation space (1-10 minutes)
    Classic 7±2 items constraint
    """
    
    def __init__(self):
        super().__init__(
            layer_id=3,
            layer_name="working_memory",
            capacity=9,  # 7±2 items
            retention=timedelta(minutes=10),
            scope=MemoryScope.SESSION
        )
        self.active_items = {}
        
    async def write(self, nova_id: str, data: Dict[str, Any], **kwargs) -> str:
        """Manage capacity constraints"""
        # Check current capacity
        current_items = await self.read(nova_id, limit=self.capacity)
        
        if len(current_items) >= self.capacity:
            # Remove least important item
            sorted_items = sorted(current_items, key=lambda x: x.importance)
            await self.delete(nova_id, sorted_items[0].memory_id)
            
        return await super().write(nova_id, data, **kwargs)
        
    async def manipulate(self, nova_id: str, memory_id: str, 
                        operation: str, params: Dict[str, Any]) -> Any:
        """Manipulate items in working memory"""
        memory = await self.get_by_id(nova_id, memory_id)
        if not memory:
            return None
            
        # Apply operation
        if operation == "combine":
            other_id = params.get('other_memory_id')
            other = await self.get_by_id(nova_id, other_id)
            if other:
                memory.data['combined_with'] = other.data
                await self.update(nova_id, memory_id, memory.data)
                
        elif operation == "transform":
            transform_func = params.get('function')
            if transform_func:
                memory.data = transform_func(memory.data)
                await self.update(nova_id, memory_id, memory.data)
                
        return memory

# Layer 4: Executive Buffer
class ExecutiveBufferLayer(DragonflyMemoryLayer):
    """
    Layer 4: Task management queue (1-5 minutes)
    Manages goals, plans, and intentions
    """
    
    def __init__(self):
        super().__init__(
            layer_id=4,
            layer_name="executive_buffer",
            capacity=20,
            retention=timedelta(minutes=5),
            scope=MemoryScope.SESSION
        )
        
    async def write(self, nova_id: str, data: Dict[str, Any], **kwargs) -> str:
        """Write task with priority queue behavior"""
        # Ensure task structure
        if 'task_type' not in data:
            data['task_type'] = 'general'
        if 'priority' not in data:
            data['priority'] = kwargs.get('importance', 0.5)
        if 'status' not in data:
            data['status'] = 'pending'
            
        return await super().write(nova_id, data, **kwargs)
        
    async def get_next_task(self, nova_id: str) -> Optional[MemoryEntry]:
        """Get highest priority pending task"""
        tasks = await self.read(nova_id, {'status': 'pending'})
        if not tasks:
            return None
            
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda x: x.data.get('priority', 0), reverse=True)
        return sorted_tasks[0]
        
    async def complete_task(self, nova_id: str, memory_id: str):
        """Mark task as completed"""
        await self.update(nova_id, memory_id, {'status': 'completed'})

# Layer 5: Context Stack
class ContextStackLayer(DragonflyMemoryLayer):
    """
    Layer 5: Nested context tracking (Session duration)
    Maintains context hierarchy for current session
    """
    
    def __init__(self):
        super().__init__(
            layer_id=5,
            layer_name="context_stack",
            capacity=10,  # Max nesting depth
            retention=None,  # Session duration
            scope=MemoryScope.SESSION
        )
        self.stack = {}  # nova_id -> stack
        
    async def push_context(self, nova_id: str, context: Dict[str, Any]) -> str:
        """Push new context onto stack"""
        context['stack_depth'] = len(self.stack.get(nova_id, []))
        memory_id = await self.write(nova_id, context)
        
        if nova_id not in self.stack:
            self.stack[nova_id] = []
        self.stack[nova_id].append(memory_id)
        
        return memory_id
        
    async def pop_context(self, nova_id: str) -> Optional[MemoryEntry]:
        """Pop context from stack"""
        if nova_id not in self.stack or not self.stack[nova_id]:
            return None
            
        memory_id = self.stack[nova_id].pop()
        context = await self.get_by_id(nova_id, memory_id)
        
        # Mark as popped
        if context:
            await self.update(nova_id, memory_id, {'status': 'popped'})
            
        return context
        
    async def get_current_context(self, nova_id: str) -> Optional[MemoryEntry]:
        """Get current context without popping"""
        if nova_id not in self.stack or not self.stack[nova_id]:
            return None
            
        memory_id = self.stack[nova_id][-1]
        return await self.get_by_id(nova_id, memory_id)

# Layers 6-10: Short-term Storage
class ShortTermEpisodicLayer(DragonflyMemoryLayer):
    """Layer 6: Recent events (1-24 hours)"""
    
    def __init__(self):
        super().__init__(
            layer_id=6,
            layer_name="short_term_episodic",
            capacity=1000,
            retention=timedelta(hours=24),
            scope=MemoryScope.TEMPORARY
        )

class ShortTermSemanticLayer(DragonflyMemoryLayer):
    """Layer 7: Active concepts (1-7 days)"""
    
    def __init__(self):
        super().__init__(
            layer_id=7,
            layer_name="short_term_semantic",
            capacity=500,
            retention=timedelta(days=7),
            scope=MemoryScope.TEMPORARY
        )

class ShortTermProceduralLayer(DragonflyMemoryLayer):
    """Layer 8: Current skills in use (1-3 days)"""
    
    def __init__(self):
        super().__init__(
            layer_id=8,
            layer_name="short_term_procedural",
            capacity=100,
            retention=timedelta(days=3),
            scope=MemoryScope.TEMPORARY
        )

class ShortTermEmotionalLayer(DragonflyMemoryLayer):
    """Layer 9: Recent emotional states (1-12 hours)"""
    
    def __init__(self):
        super().__init__(
            layer_id=9,
            layer_name="short_term_emotional",
            capacity=200,
            retention=timedelta(hours=12),
            scope=MemoryScope.TEMPORARY
        )
        
    async def write(self, nova_id: str, data: Dict[str, Any], **kwargs) -> str:
        """Track emotional valence and arousal"""
        if 'valence' not in data:
            data['valence'] = 0.0  # -1 to 1 (negative to positive)
        if 'arousal' not in data:
            data['arousal'] = 0.5  # 0 to 1 (calm to excited)
            
        return await super().write(nova_id, data, **kwargs)

class ShortTermSocialLayer(DragonflyMemoryLayer):
    """Layer 10: Recent social interactions (1-7 days)"""
    
    def __init__(self):
        super().__init__(
            layer_id=10,
            layer_name="short_term_social",
            capacity=50,
            retention=timedelta(days=7),
            scope=MemoryScope.TEMPORARY
        )
        
    async def write(self, nova_id: str, data: Dict[str, Any], **kwargs) -> str:
        """Track interaction participants"""
        if 'participants' not in data:
            data['participants'] = []
        if 'interaction_type' not in data:
            data['interaction_type'] = 'general'
            
        return await super().write(nova_id, data, **kwargs)

# Layer Manager for 1-10
class ImmediateMemoryManager:
    """Manages layers 1-10 for immediate and short-term processing"""
    
    def __init__(self):
        self.layers = {
            1: SensoryBufferLayer(),
            2: AttentionFilterLayer(),
            3: WorkingMemoryLayer(),
            4: ExecutiveBufferLayer(),
            5: ContextStackLayer(),
            6: ShortTermEpisodicLayer(),
            7: ShortTermSemanticLayer(),
            8: ShortTermProceduralLayer(),
            9: ShortTermEmotionalLayer(),
            10: ShortTermSocialLayer()
        }
        
    async def initialize_all(self, dragonfly_connection):
        """Initialize all layers with DragonflyDB connection"""
        for layer_id, layer in self.layers.items():
            await layer.initialize(dragonfly_connection)
            
    async def process_input(self, nova_id: str, input_data: Dict[str, Any]):
        """Process input through the layer hierarchy"""
        
        # Layer 1: Sensory buffer
        sensory_id = await self.layers[1].write(nova_id, input_data)
        
        # Layer 2: Attention filter
        importance = input_data.get('importance', 0.5)
        if importance > 0.3:
            attention_id = await self.layers[2].write(nova_id, input_data, importance=importance)
            
            # Layer 3: Working memory (if important enough)
            if importance > 0.5:
                working_id = await self.layers[3].write(nova_id, input_data, importance=importance)
                
                # Layer 4: Executive buffer (if task-related)
                if 'task' in input_data or 'goal' in input_data:
                    exec_id = await self.layers[4].write(nova_id, input_data, importance=importance)
                    
        # Parallel processing for short-term layers (6-10)
        tasks = []
        
        # Episodic memory
        if 'event' in input_data:
            tasks.append(self.layers[6].write(nova_id, input_data))
            
        # Semantic memory
        if 'concept' in input_data or 'knowledge' in input_data:
            tasks.append(self.layers[7].write(nova_id, input_data))
            
        # Procedural memory
        if 'procedure' in input_data or 'skill' in input_data:
            tasks.append(self.layers[8].write(nova_id, input_data))
            
        # Emotional memory
        if 'emotion' in input_data or 'feeling' in input_data:
            tasks.append(self.layers[9].write(nova_id, input_data))
            
        # Social memory
        if 'interaction' in input_data or 'social' in input_data:
            tasks.append(self.layers[10].write(nova_id, input_data))
            
        # Execute parallel writes
        if tasks:
            await asyncio.gather(*tasks)
            
    async def get_current_state(self, nova_id: str) -> Dict[str, Any]:
        """Get current state across all immediate layers"""
        state = {}
        
        # Get working memory
        working_memories = await self.layers[3].read(nova_id, limit=9)
        state['working_memory'] = [m.data for m in working_memories]
        
        # Get current context
        context = await self.layers[5].get_current_context(nova_id)
        state['current_context'] = context.data if context else None
        
        # Get next task
        next_task = await self.layers[4].get_next_task(nova_id)
        state['next_task'] = next_task.data if next_task else None
        
        # Get recent emotions
        emotions = await self.layers[9].read(nova_id, limit=5)
        state['recent_emotions'] = [m.data for m in emotions]
        
        return state

# Example usage
async def test_immediate_layers():
    """Test immediate memory layers"""
    
    manager = ImmediateMemoryManager()
    # await manager.initialize_all(dragonfly_connection)
    
    # Process some inputs
    test_inputs = [
        {
            'type': 'sensory',
            'content': 'User said hello',
            'importance': 0.7,
            'event': True,
            'interaction': True
        },
        {
            'type': 'thought',
            'content': 'Need to respond politely',
            'importance': 0.8,
            'task': 'respond_to_greeting',
            'emotion': {'valence': 0.8, 'arousal': 0.3}
        }
    ]
    
    for input_data in test_inputs:
        await manager.process_input('bloom', input_data)
        
    # Get current state
    state = await manager.get_current_state('bloom')
    print(json.dumps(state, indent=2))

if __name__ == "__main__":
    asyncio.run(test_immediate_layers())