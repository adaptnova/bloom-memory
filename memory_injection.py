#!/usr/bin/env python3
"""
Nova Memory System - Session Memory Injection
Handles memory loading strategies for Nova consciousness startup
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from unified_memory_api import NovaMemoryAPI, MemoryType
from memory_layers import MemoryEntry, MemoryImportance

logger = logging.getLogger(__name__)

class InjectionMode(Enum):
    """Memory injection modes for session startup"""
    CONTINUE = "continue"      # Resume from last state
    RESUME = "resume"          # Resume from specific checkpoint
    COMPACT = "compact"        # Load compressed summary
    FRESH = "fresh"           # Clean start with identity only
    SELECTIVE = "selective"    # Load specific memory types
    RECOVERY = "recovery"      # Recovery from corruption

@dataclass
class InjectionProfile:
    """Configuration for memory injection"""
    mode: InjectionMode
    nova_id: str
    session_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    time_window: Optional[timedelta] = None
    memory_types: Optional[List[MemoryType]] = None
    importance_threshold: float = 0.3
    max_memories: int = 1000
    
class MemoryInjector:
    """
    Handles memory injection for Nova session startup
    Optimizes what memories to load based on mode and context
    """
    
    def __init__(self, memory_api: NovaMemoryAPI):
        self.memory_api = memory_api
        self.injection_strategies = {
            InjectionMode.CONTINUE: self._inject_continue,
            InjectionMode.RESUME: self._inject_resume,
            InjectionMode.COMPACT: self._inject_compact,
            InjectionMode.FRESH: self._inject_fresh,
            InjectionMode.SELECTIVE: self._inject_selective,
            InjectionMode.RECOVERY: self._inject_recovery
        }
        
    async def inject_memory(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Main entry point for memory injection
        Returns injection summary and statistics
        """
        logger.info(f"Starting memory injection for {profile.nova_id} in {profile.mode.value} mode")
        
        start_time = datetime.now()
        
        # Get injection strategy
        strategy = self.injection_strategies.get(profile.mode)
        if not strategy:
            raise ValueError(f"Unknown injection mode: {profile.mode}")
            
        # Execute injection
        result = await strategy(profile)
        
        # Calculate statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result['statistics'] = {
            'injection_mode': profile.mode.value,
            'duration_seconds': duration,
            'timestamp': end_time.isoformat()
        }
        
        logger.info(f"Memory injection completed in {duration:.2f} seconds")
        
        return result
        
    async def _inject_continue(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Continue mode: Load recent memories from all layers
        Best for resuming after short breaks
        """
        result = {
            'mode': 'continue',
            'loaded_memories': {},
            'layer_summary': {}
        }
        
        # Define time windows for different memory types
        time_windows = {
            MemoryType.WORKING: timedelta(minutes=10),
            MemoryType.ATTENTION: timedelta(minutes=30),
            MemoryType.TASK: timedelta(hours=1),
            MemoryType.CONTEXT: timedelta(hours=2),
            MemoryType.EPISODIC: timedelta(hours=24),
            MemoryType.EMOTIONAL: timedelta(hours=12),
            MemoryType.SOCIAL: timedelta(days=7)
        }
        
        # Load memories by type
        for memory_type, window in time_windows.items():
            response = await self.memory_api.recall(
                profile.nova_id,
                memory_types=[memory_type],
                time_range=window,
                limit=100
            )
            
            if response.success:
                memories = response.data.get('memories', [])
                result['loaded_memories'][memory_type.value] = len(memories)
                
                # Load into appropriate layers
                for memory in memories:
                    await self._reinject_memory(profile.nova_id, memory)
                    
        # Load working memory (most recent items)
        working_response = await self.memory_api.recall(
            profile.nova_id,
            memory_types=[MemoryType.WORKING],
            limit=9  # 7±2 constraint
        )
        
        if working_response.success:
            result['working_memory_restored'] = len(working_response.data.get('memories', []))
            
        # Get current context stack
        context_response = await self.memory_api.recall(
            profile.nova_id,
            memory_types=[MemoryType.CONTEXT],
            limit=10
        )
        
        if context_response.success:
            result['context_stack_depth'] = len(context_response.data.get('memories', []))
            
        return result
        
    async def _inject_resume(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Resume mode: Load from specific checkpoint
        Best for resuming specific work sessions
        """
        result = {
            'mode': 'resume',
            'checkpoint_id': profile.checkpoint_id,
            'loaded_memories': {}
        }
        
        if not profile.checkpoint_id:
            # Find most recent checkpoint
            checkpoints = await self._find_checkpoints(profile.nova_id)
            if checkpoints:
                profile.checkpoint_id = checkpoints[0]['checkpoint_id']
                
        if profile.checkpoint_id:
            # Load checkpoint data
            checkpoint_data = await self._load_checkpoint(profile.nova_id, profile.checkpoint_id)
            
            if checkpoint_data:
                # Restore memory state from checkpoint
                for layer_name, memories in checkpoint_data.get('memory_state', {}).items():
                    result['loaded_memories'][layer_name] = len(memories)
                    
                    for memory in memories:
                        await self._reinject_memory(profile.nova_id, memory)
                        
                result['checkpoint_loaded'] = True
                result['checkpoint_timestamp'] = checkpoint_data.get('timestamp')
            else:
                result['checkpoint_loaded'] = False
                
        return result
        
    async def _inject_compact(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Compact mode: Load compressed memory summaries
        Best for resource-constrained startups
        """
        result = {
            'mode': 'compact',
            'loaded_summaries': {}
        }
        
        # Priority memory types for compact mode
        priority_types = [
            MemoryType.WORKING,
            MemoryType.TASK,
            MemoryType.CONTEXT,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL
        ]
        
        for memory_type in priority_types:
            # Get high-importance memories only
            response = await self.memory_api.recall(
                profile.nova_id,
                memory_types=[memory_type],
                limit=20  # Fewer memories in compact mode
            )
            
            if response.success:
                memories = response.data.get('memories', [])
                
                # Filter by importance
                important_memories = [
                    m for m in memories 
                    if m.get('importance', 0) >= profile.importance_threshold
                ]
                
                result['loaded_summaries'][memory_type.value] = len(important_memories)
                
                # Create summary entries
                for memory in important_memories:
                    summary = self._create_memory_summary(memory)
                    await self._reinject_memory(profile.nova_id, summary)
                    
        # Load identity core
        identity_response = await self.memory_api.recall(
            profile.nova_id,
            query={'layer_name': 'identity_memory'},
            limit=10
        )
        
        if identity_response.success:
            result['identity_core_loaded'] = True
            
        return result
        
    async def _inject_fresh(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Fresh mode: Clean start with only identity
        Best for new sessions or testing
        """
        result = {
            'mode': 'fresh',
            'loaded_components': []
        }
        
        # Load only identity and core configuration
        identity_response = await self.memory_api.recall(
            profile.nova_id,
            query={'layer_name': 'identity_memory'},
            limit=10
        )
        
        if identity_response.success:
            result['loaded_components'].append('identity')
            
        # Load core procedural knowledge
        procedures_response = await self.memory_api.recall(
            profile.nova_id,
            memory_types=[MemoryType.PROCEDURAL],
            query={'importance_gte': 0.8},  # Only critical procedures
            limit=10
        )
        
        if procedures_response.success:
            result['loaded_components'].append('core_procedures')
            result['procedures_loaded'] = len(procedures_response.data.get('memories', []))
            
        # Initialize empty working memory
        await self.memory_api.remember(
            profile.nova_id,
            {'initialized': True, 'mode': 'fresh'},
            memory_type=MemoryType.WORKING,
            importance=0.1
        )
        
        result['working_memory_initialized'] = True
        
        return result
        
    async def _inject_selective(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Selective mode: Load specific memory types
        Best for specialized operations
        """
        result = {
            'mode': 'selective',
            'requested_types': [mt.value for mt in (profile.memory_types or [])],
            'loaded_memories': {}
        }
        
        if not profile.memory_types:
            profile.memory_types = [MemoryType.WORKING, MemoryType.SEMANTIC]
            
        for memory_type in profile.memory_types:
            response = await self.memory_api.recall(
                profile.nova_id,
                memory_types=[memory_type],
                time_range=profile.time_window,
                limit=profile.max_memories // len(profile.memory_types)
            )
            
            if response.success:
                memories = response.data.get('memories', [])
                result['loaded_memories'][memory_type.value] = len(memories)
                
                for memory in memories:
                    await self._reinject_memory(profile.nova_id, memory)
                    
        return result
        
    async def _inject_recovery(self, profile: InjectionProfile) -> Dict[str, Any]:
        """
        Recovery mode: Attempt to recover from corruption
        Best for error recovery scenarios
        """
        result = {
            'mode': 'recovery',
            'recovery_attempts': {},
            'recovered_memories': 0
        }
        
        # Try to recover from each database
        databases = ['dragonfly', 'postgresql', 'couchdb', 'arangodb']
        
        for db in databases:
            try:
                # Attempt to read from each database
                response = await self.memory_api.recall(
                    profile.nova_id,
                    query={'database': db},
                    limit=100
                )
                
                if response.success:
                    memories = response.data.get('memories', [])
                    result['recovery_attempts'][db] = {
                        'success': True,
                        'recovered': len(memories)
                    }
                    result['recovered_memories'] += len(memories)
                    
                    # Reinject recovered memories
                    for memory in memories:
                        await self._reinject_memory(profile.nova_id, memory, safe_mode=True)
                        
            except Exception as e:
                result['recovery_attempts'][db] = {
                    'success': False,
                    'error': str(e)
                }
                
        # Attempt checkpoint recovery
        checkpoints = await self._find_checkpoints(profile.nova_id)
        if checkpoints:
            result['checkpoints_found'] = len(checkpoints)
            # Use most recent valid checkpoint
            for checkpoint in checkpoints:
                if await self._validate_checkpoint(checkpoint):
                    result['checkpoint_recovery'] = checkpoint['checkpoint_id']
                    break
                    
        return result
        
    async def _reinject_memory(self, nova_id: str, memory: Dict[str, Any], 
                              safe_mode: bool = False) -> bool:
        """Reinject a memory into the appropriate layer"""
        try:
            # Extract memory data
            content = memory.get('data', memory.get('content', {}))
            importance = memory.get('importance', 0.5)
            context = memory.get('context', 'reinjected')
            memory_type = memory.get('memory_type')
            
            # Add reinjection metadata
            if isinstance(content, dict):
                content['reinjected'] = True
                content['original_timestamp'] = memory.get('timestamp')
                
            # Write to memory system
            response = await self.memory_api.remember(
                nova_id,
                content,
                importance=importance,
                context=context,
                memory_type=MemoryType(memory_type) if memory_type else None
            )
            
            return response.success
            
        except Exception as e:
            if not safe_mode:
                raise
            logger.warning(f"Failed to reinject memory: {e}")
            return False
            
    def _create_memory_summary(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compressed summary of a memory"""
        summary = {
            'summary': True,
            'original_id': memory.get('memory_id'),
            'timestamp': memory.get('timestamp'),
            'importance': memory.get('importance', 0.5),
            'type': memory.get('memory_type', 'unknown')
        }
        
        # Extract key information
        data = memory.get('data', {})
        if isinstance(data, dict):
            # Keep only important fields
            important_fields = ['content', 'task', 'goal', 'concept', 'emotion', 'result']
            summary['key_data'] = {
                k: v for k, v in data.items() 
                if k in important_fields
            }
        else:
            summary['key_data'] = {'content': str(data)[:100]}  # Truncate
            
        return summary
        
    async def _find_checkpoints(self, nova_id: str) -> List[Dict[str, Any]]:
        """Find available checkpoints for a Nova"""
        # This would query checkpoint storage
        # For now, return empty list
        return []
        
    async def _load_checkpoint(self, nova_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint"""
        # This would load from checkpoint storage
        # For now, return None
        return None
        
    async def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Validate checkpoint integrity"""
        # Check required fields
        required = ['checkpoint_id', 'timestamp', 'memory_state']
        return all(field in checkpoint for field in required)

class MemoryCompactor:
    """
    Handles memory compaction for long-term storage
    Reduces memory footprint while preserving important information
    """
    
    def __init__(self, memory_api: NovaMemoryAPI):
        self.memory_api = memory_api
        self.compaction_rules = {
            'age_threshold': timedelta(days=7),
            'importance_threshold': 0.3,
            'compression_ratio': 0.2,  # Keep 20% of memories
            'preserve_types': [MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
        }
        
    async def compact_memories(self, nova_id: str, aggressive: bool = False) -> Dict[str, Any]:
        """
        Compact memories based on age, importance, and type
        """
        result = {
            'compacted': 0,
            'preserved': 0,
            'deleted': 0,
            'space_saved': 0
        }
        
        # Adjust rules for aggressive mode
        if aggressive:
            self.compaction_rules['compression_ratio'] = 0.1
            self.compaction_rules['importance_threshold'] = 0.5
            
        # Get all memories older than threshold
        cutoff_time = datetime.now() - self.compaction_rules['age_threshold']
        
        response = await self.memory_api.recall(
            nova_id,
            query={'before': cutoff_time.isoformat()},
            limit=10000
        )
        
        if not response.success:
            return result
            
        memories = response.data.get('memories', [])
        
        # Sort by importance
        memories.sort(key=lambda m: m.get('importance', 0), reverse=True)
        
        # Determine how many to keep
        keep_count = int(len(memories) * self.compaction_rules['compression_ratio'])
        
        # Process memories
        for i, memory in enumerate(memories):
            memory_type = memory.get('memory_type')
            importance = memory.get('importance', 0)
            
            # Preserve certain types
            if memory_type in [mt.value for mt in self.compaction_rules['preserve_types']]:
                result['preserved'] += 1
                continue
                
            # Keep high importance
            if importance >= self.compaction_rules['importance_threshold']:
                result['preserved'] += 1
                continue
                
            # Keep top N
            if i < keep_count:
                # Compact but keep
                compacted = await self._compact_memory(nova_id, memory)
                if compacted:
                    result['compacted'] += 1
            else:
                # Delete
                deleted = await self._delete_memory(nova_id, memory)
                if deleted:
                    result['deleted'] += 1
                    
        # Calculate space saved (simplified)
        result['space_saved'] = result['deleted'] * 1024  # Assume 1KB per memory
        
        return result
        
    async def _compact_memory(self, nova_id: str, memory: Dict[str, Any]) -> bool:
        """Compact a single memory"""
        # Create summary
        summary = {
            'compacted': True,
            'original_id': memory.get('memory_id'),
            'timestamp': memory.get('timestamp'),
            'importance': memory.get('importance'),
            'summary': self._generate_summary(memory.get('data', {}))
        }
        
        # Update memory with compacted version
        response = await self.memory_api.execute(MemoryRequest(
            operation=MemoryOperation.UPDATE,
            nova_id=nova_id,
            query={'memory_id': memory.get('memory_id')},
            data=summary
        ))
        
        return response.success
        
    async def _delete_memory(self, nova_id: str, memory: Dict[str, Any]) -> bool:
        """Delete a memory"""
        response = await self.memory_api.execute(MemoryRequest(
            operation=MemoryOperation.DELETE,
            nova_id=nova_id,
            query={'memory_id': memory.get('memory_id')}
        ))
        
        return response.success
        
    def _generate_summary(self, data: Any) -> str:
        """Generate text summary of memory data"""
        if isinstance(data, dict):
            # Extract key information
            key_parts = []
            for k, v in data.items():
                if k in ['content', 'task', 'concept', 'result']:
                    key_parts.append(f"{k}:{str(v)[:50]}")
            return "; ".join(key_parts)
        else:
            return str(data)[:100]

# Example usage
async def test_memory_injection():
    """Test memory injection system"""
    
    # Initialize API
    api = NovaMemoryAPI()
    await api.initialize()
    
    # Create injector
    injector = MemoryInjector(api)
    
    # Test different injection modes
    
    # Continue mode
    print("\n=== Testing CONTINUE mode ===")
    profile = InjectionProfile(
        mode=InjectionMode.CONTINUE,
        nova_id='bloom'
    )
    result = await injector.inject_memory(profile)
    print(json.dumps(result, indent=2))
    
    # Compact mode
    print("\n=== Testing COMPACT mode ===")
    profile = InjectionProfile(
        mode=InjectionMode.COMPACT,
        nova_id='bloom',
        importance_threshold=0.7
    )
    result = await injector.inject_memory(profile)
    print(json.dumps(result, indent=2))
    
    # Fresh mode
    print("\n=== Testing FRESH mode ===")
    profile = InjectionProfile(
        mode=InjectionMode.FRESH,
        nova_id='bloom'
    )
    result = await injector.inject_memory(profile)
    print(json.dumps(result, indent=2))
    
    # Test compactor
    print("\n=== Testing Memory Compaction ===")
    compactor = MemoryCompactor(api)
    compact_result = await compactor.compact_memories('bloom', aggressive=False)
    print(json.dumps(compact_result, indent=2))
    
    await api.shutdown()

if __name__ == "__main__":
    asyncio.run(test_memory_injection())