#!/usr/bin/env python3
"""
Nova Memory System - Consolidation Engine
Manages memory flow from short-term to long-term storage
Implements sleep-like consolidation cycles
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from unified_memory_api import NovaMemoryAPI, MemoryType
from database_connections import NovaDatabasePool
from postgresql_memory_layer import (
    EpisodicConsolidationLayer, SemanticIntegrationLayer,
    ProceduralCompilationLayer, LongTermEpisodicLayer
)
from couchdb_memory_layer import (
    SemanticMemoryLayer, CreativeMemoryLayer, NarrativeMemoryLayer
)

logger = logging.getLogger(__name__)

class ConsolidationPhase(Enum):
    """Memory consolidation phases (inspired by sleep cycles)"""
    ACTIVE = "active"           # Normal waking state
    QUIET = "quiet"             # Initial consolidation
    SLOW_WAVE = "slow_wave"     # Deep consolidation
    REM = "rem"                 # Creative consolidation
    INTEGRATION = "integration"  # Final integration

@dataclass
class ConsolidationCycle:
    """Single consolidation cycle configuration"""
    phase: ConsolidationPhase
    duration: timedelta
    memory_types: List[MemoryType]
    consolidation_rate: float  # 0.0 to 1.0
    importance_threshold: float
    
class MemoryConsolidationEngine:
    """
    Manages the complex process of memory consolidation
    Inspired by human sleep cycles and memory formation
    """
    
    def __init__(self, memory_api: NovaMemoryAPI, db_pool: NovaDatabasePool):
        self.memory_api = memory_api
        self.db_pool = db_pool
        
        # Initialize consolidation layers
        self.consolidation_layers = {
            'episodic': EpisodicConsolidationLayer(),
            'semantic': SemanticIntegrationLayer(),
            'procedural': ProceduralCompilationLayer(),
            'long_term_episodic': LongTermEpisodicLayer(),
            'semantic_knowledge': SemanticMemoryLayer(),
            'creative': CreativeMemoryLayer(),
            'narrative': NarrativeMemoryLayer()
        }
        
        # Consolidation cycles configuration
        self.cycles = [
            ConsolidationCycle(
                phase=ConsolidationPhase.QUIET,
                duration=timedelta(minutes=30),
                memory_types=[MemoryType.EPISODIC, MemoryType.SOCIAL],
                consolidation_rate=0.3,
                importance_threshold=0.4
            ),
            ConsolidationCycle(
                phase=ConsolidationPhase.SLOW_WAVE,
                duration=timedelta(minutes=45),
                memory_types=[MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
                consolidation_rate=0.5,
                importance_threshold=0.5
            ),
            ConsolidationCycle(
                phase=ConsolidationPhase.REM,
                duration=timedelta(minutes=20),
                memory_types=[MemoryType.EMOTIONAL, MemoryType.CREATIVE],
                consolidation_rate=0.2,
                importance_threshold=0.3
            ),
            ConsolidationCycle(
                phase=ConsolidationPhase.INTEGRATION,
                duration=timedelta(minutes=15),
                memory_types=[MemoryType.METACOGNITIVE, MemoryType.PREDICTIVE],
                consolidation_rate=0.7,
                importance_threshold=0.6
            )
        ]
        
        self.current_phase = ConsolidationPhase.ACTIVE
        self.consolidation_stats = {
            'total_consolidated': 0,
            'patterns_discovered': 0,
            'memories_compressed': 0,
            'creative_insights': 0
        }
        
        self.is_running = False
        self.consolidation_task = None
        
    async def initialize(self):
        """Initialize all consolidation layers"""
        # Initialize PostgreSQL layers
        pg_conn = self.db_pool.get_connection('postgresql')
        for layer_name in ['episodic', 'semantic', 'procedural', 'long_term_episodic']:
            await self.consolidation_layers[layer_name].initialize(pg_conn)
            
        # Initialize CouchDB layers
        couch_conn = self.db_pool.get_connection('couchdb')
        for layer_name in ['semantic_knowledge', 'creative', 'narrative']:
            await self.consolidation_layers[layer_name].initialize(couch_conn)
            
        logger.info("Consolidation engine initialized")
        
    async def start_automatic_consolidation(self, nova_id: str):
        """Start automatic consolidation cycles"""
        if self.is_running:
            logger.warning("Consolidation already running")
            return
            
        self.is_running = True
        self.consolidation_task = asyncio.create_task(
            self._run_consolidation_cycles(nova_id)
        )
        logger.info(f"Started automatic consolidation for {nova_id}")
        
    async def stop_automatic_consolidation(self):
        """Stop automatic consolidation"""
        self.is_running = False
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped automatic consolidation")
        
    async def _run_consolidation_cycles(self, nova_id: str):
        """Run continuous consolidation cycles"""
        cycle_index = 0
        
        while self.is_running:
            try:
                # Get current cycle
                cycle = self.cycles[cycle_index % len(self.cycles)]
                self.current_phase = cycle.phase
                
                logger.info(f"Starting {cycle.phase.value} consolidation phase")
                
                # Run consolidation for this cycle
                await self._consolidate_cycle(nova_id, cycle)
                
                # Wait for cycle duration
                await asyncio.sleep(cycle.duration.total_seconds())
                
                # Move to next cycle
                cycle_index += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
                
    async def _consolidate_cycle(self, nova_id: str, cycle: ConsolidationCycle):
        """Execute single consolidation cycle"""
        start_time = datetime.now()
        
        # Get memories for consolidation
        memories_to_consolidate = await self._select_memories_for_consolidation(
            nova_id, cycle
        )
        
        consolidated_count = 0
        
        for memory_batch in self._batch_memories(memories_to_consolidate, 100):
            if not self.is_running:
                break
                
            # Process based on phase
            if cycle.phase == ConsolidationPhase.QUIET:
                consolidated_count += await self._quiet_consolidation(nova_id, memory_batch)
                
            elif cycle.phase == ConsolidationPhase.SLOW_WAVE:
                consolidated_count += await self._slow_wave_consolidation(nova_id, memory_batch)
                
            elif cycle.phase == ConsolidationPhase.REM:
                consolidated_count += await self._rem_consolidation(nova_id, memory_batch)
                
            elif cycle.phase == ConsolidationPhase.INTEGRATION:
                consolidated_count += await self._integration_consolidation(nova_id, memory_batch)
                
        # Update statistics
        self.consolidation_stats['total_consolidated'] += consolidated_count
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Consolidated {consolidated_count} memories in {duration:.2f}s")
        
    async def _select_memories_for_consolidation(self, nova_id: str, 
                                                cycle: ConsolidationCycle) -> List[Dict]:
        """Select appropriate memories for consolidation"""
        memories = []
        
        # Query memories based on cycle configuration
        for memory_type in cycle.memory_types:
            response = await self.memory_api.recall(
                nova_id,
                memory_types=[memory_type],
                time_range=timedelta(hours=24),  # Last 24 hours
                limit=1000
            )
            
            if response.success:
                # Filter by importance and consolidation status
                for memory in response.data.get('memories', []):
                    if (memory.get('importance', 0) >= cycle.importance_threshold and
                        not memory.get('consolidated', False)):
                        memories.append(memory)
                        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.get('importance', 0), m.get('timestamp', '')), 
                     reverse=True)
        
        # Apply consolidation rate
        max_to_consolidate = int(len(memories) * cycle.consolidation_rate)
        return memories[:max_to_consolidate]
        
    def _batch_memories(self, memories: List[Dict], batch_size: int):
        """Yield memories in batches"""
        for i in range(0, len(memories), batch_size):
            yield memories[i:i + batch_size]
            
    async def _quiet_consolidation(self, nova_id: str, memories: List[Dict]) -> int:
        """
        Quiet consolidation: Initial filtering and organization
        Focus on episodic and social memories
        """
        consolidated = 0
        
        # Group by context
        context_groups = {}
        for memory in memories:
            context = memory.get('context', 'general')
            if context not in context_groups:
                context_groups[context] = []
            context_groups[context].append(memory)
            
        # Consolidate each context group
        for context, group_memories in context_groups.items():
            if len(group_memories) > 5:  # Only consolidate if enough memories
                # Create consolidated episode
                consolidated_episode = {
                    'type': 'consolidated_episode',
                    'context': context,
                    'memories': [self._summarize_memory(m) for m in group_memories],
                    'time_span': {
                        'start': min(m.get('timestamp', '') for m in group_memories),
                        'end': max(m.get('timestamp', '') for m in group_memories)
                    },
                    'total_importance': sum(m.get('importance', 0) for m in group_memories)
                }
                
                # Write to episodic consolidation layer
                await self.consolidation_layers['episodic'].write(
                    nova_id,
                    consolidated_episode,
                    importance=consolidated_episode['total_importance'] / len(group_memories),
                    context=f'consolidated_{context}'
                )
                
                consolidated += len(group_memories)
                
        return consolidated
        
    async def _slow_wave_consolidation(self, nova_id: str, memories: List[Dict]) -> int:
        """
        Slow wave consolidation: Deep processing and integration
        Focus on semantic and procedural memories
        """
        consolidated = 0
        
        # Extract concepts and procedures
        concepts = []
        procedures = []
        
        for memory in memories:
            data = memory.get('data', {})
            
            # Identify concepts
            if any(key in data for key in ['concept', 'knowledge', 'definition']):
                concepts.append(memory)
                
            # Identify procedures
            elif any(key in data for key in ['procedure', 'steps', 'method']):
                procedures.append(memory)
                
        # Consolidate concepts into semantic knowledge
        if concepts:
            # Find relationships between concepts
            concept_graph = await self._build_concept_relationships(concepts)
            
            # Store integrated knowledge
            await self.consolidation_layers['semantic'].integrate_concepts(
                nova_id, 
                [self._extract_concept(c) for c in concepts]
            )
            
            consolidated += len(concepts)
            
        # Compile procedures
        if procedures:
            # Group similar procedures
            procedure_groups = self._group_similar_procedures(procedures)
            
            for group_name, group_procedures in procedure_groups.items():
                # Compile into optimized procedure
                await self.consolidation_layers['procedural'].compile_procedure(
                    nova_id,
                    [self._extract_steps(p) for p in group_procedures],
                    group_name
                )
                
            consolidated += len(procedures)
            
        return consolidated
        
    async def _rem_consolidation(self, nova_id: str, memories: List[Dict]) -> int:
        """
        REM consolidation: Creative combinations and emotional processing
        Focus on emotional and creative insights
        """
        consolidated = 0
        
        # Extract emotional patterns
        emotional_memories = [m for m in memories 
                            if m.get('data', {}).get('emotion') or 
                               m.get('context') == 'emotional']
        
        if emotional_memories:
            # Analyze emotional patterns
            emotional_patterns = self._analyze_emotional_patterns(emotional_memories)
            
            # Store patterns
            for pattern in emotional_patterns:
                await self.consolidation_layers['long_term_episodic'].write(
                    nova_id,
                    pattern,
                    importance=0.7,
                    context='emotional_pattern'
                )
                
            self.consolidation_stats['patterns_discovered'] += len(emotional_patterns)
            
        # Generate creative combinations
        if len(memories) >= 3:
            # Random sampling for creative combinations
            import random
            sample_size = min(10, len(memories))
            sampled = random.sample(memories, sample_size)
            
            # Create novel combinations
            combinations = await self._generate_creative_combinations(sampled)
            
            for combination in combinations:
                await self.consolidation_layers['creative'].create_combination(
                    nova_id,
                    combination['elements'],
                    combination['type']
                )
                
            self.consolidation_stats['creative_insights'] += len(combinations)
            consolidated += len(combinations)
            
        # Create narratives from episodic sequences
        if len(memories) > 5:
            narrative = self._construct_narrative(memories)
            if narrative:
                await self.consolidation_layers['narrative'].store_narrative(
                    nova_id,
                    narrative,
                    'consolidated_experience'
                )
                consolidated += 1
                
        return consolidated
        
    async def _integration_consolidation(self, nova_id: str, memories: List[Dict]) -> int:
        """
        Integration consolidation: Meta-cognitive processing
        Focus on patterns, predictions, and system optimization
        """
        consolidated = 0
        
        # Analyze memory patterns
        patterns = await self._analyze_memory_patterns(nova_id, memories)
        
        # Store meta-cognitive insights
        for pattern in patterns:
            await self.memory_api.remember(
                nova_id,
                pattern,
                memory_type=MemoryType.METACOGNITIVE,
                importance=0.8,
                context='pattern_recognition'
            )
            
        # Generate predictions based on patterns
        predictions = self._generate_predictions(patterns)
        
        for prediction in predictions:
            await self.memory_api.remember(
                nova_id,
                prediction,
                memory_type=MemoryType.PREDICTIVE,
                importance=0.7,
                context='future_projection'
            )
            
        # Optimize memory organization
        optimization_suggestions = self._suggest_optimizations(memories)
        
        if optimization_suggestions:
            await self.memory_api.remember(
                nova_id,
                {
                    'type': 'memory_optimization',
                    'suggestions': optimization_suggestions,
                    'timestamp': datetime.now().isoformat()
                },
                memory_type=MemoryType.METACOGNITIVE,
                importance=0.9
            )
            
        consolidated += len(patterns) + len(predictions)
        return consolidated
        
    def _summarize_memory(self, memory: Dict) -> Dict:
        """Create summary of memory for consolidation"""
        return {
            'id': memory.get('memory_id'),
            'key_content': str(memory.get('data', {}))[:100],
            'importance': memory.get('importance', 0.5),
            'timestamp': memory.get('timestamp')
        }
        
    def _extract_concept(self, memory: Dict) -> Dict:
        """Extract concept information from memory"""
        data = memory.get('data', {})
        return {
            'concept': data.get('concept', data.get('content', 'unknown')),
            'definition': data.get('definition', data.get('knowledge', {})),
            'source': memory.get('context', 'general'),
            'confidence': memory.get('importance', 0.5)
        }
        
    def _extract_steps(self, memory: Dict) -> List[Dict]:
        """Extract procedural steps from memory"""
        data = memory.get('data', {})
        
        if 'steps' in data:
            return data['steps']
        elif 'procedure' in data:
            # Convert procedure to steps
            return [{'action': data['procedure'], 'order': 1}]
        else:
            return [{'action': str(data), 'order': 1}]
            
    async def _build_concept_relationships(self, concepts: List[Dict]) -> Dict:
        """Build relationships between concepts"""
        relationships = []
        
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Simple similarity check
                c1_text = str(concept1.get('data', {})).lower()
                c2_text = str(concept2.get('data', {})).lower()
                
                # Check for common words
                words1 = set(c1_text.split())
                words2 = set(c2_text.split())
                common = words1.intersection(words2)
                
                if len(common) > 2:  # At least 2 common words
                    relationships.append({
                        'from': concept1.get('memory_id'),
                        'to': concept2.get('memory_id'),
                        'type': 'related',
                        'strength': len(common) / max(len(words1), len(words2))
                    })
                    
        return {'concepts': concepts, 'relationships': relationships}
        
    def _group_similar_procedures(self, procedures: List[Dict]) -> Dict[str, List[Dict]]:
        """Group similar procedures together"""
        groups = {}
        
        for procedure in procedures:
            # Simple grouping by first action word
            data = procedure.get('data', {})
            action = str(data.get('procedure', data.get('action', 'unknown')))
            
            key = action.split()[0] if action else 'misc'
            if key not in groups:
                groups[key] = []
            groups[key].append(procedure)
            
        return groups
        
    def _analyze_emotional_patterns(self, memories: List[Dict]) -> List[Dict]:
        """Analyze patterns in emotional memories"""
        patterns = []
        
        # Group by emotion type
        emotion_groups = {}
        for memory in memories:
            emotion = memory.get('data', {}).get('emotion', {})
            emotion_type = emotion.get('type', 'unknown')
            
            if emotion_type not in emotion_groups:
                emotion_groups[emotion_type] = []
            emotion_groups[emotion_type].append(memory)
            
        # Find patterns in each group
        for emotion_type, group in emotion_groups.items():
            if len(group) > 3:
                # Calculate average valence and arousal
                valences = [m.get('data', {}).get('emotion', {}).get('valence', 0) 
                           for m in group]
                arousals = [m.get('data', {}).get('emotion', {}).get('arousal', 0.5) 
                           for m in group]
                
                pattern = {
                    'pattern_type': 'emotional_tendency',
                    'emotion': emotion_type,
                    'frequency': len(group),
                    'average_valence': np.mean(valences),
                    'average_arousal': np.mean(arousals),
                    'triggers': self._extract_triggers(group)
                }
                
                patterns.append(pattern)
                
        return patterns
        
    def _extract_triggers(self, emotional_memories: List[Dict]) -> List[str]:
        """Extract common triggers from emotional memories"""
        triggers = []
        
        for memory in emotional_memories:
            context = memory.get('context', '')
            if context and context != 'general':
                triggers.append(context)
                
        # Return unique triggers
        return list(set(triggers))
        
    async def _generate_creative_combinations(self, memories: List[Dict]) -> List[Dict]:
        """Generate creative combinations from memories"""
        combinations = []
        
        # Try different combination strategies
        if len(memories) >= 2:
            # Analogical combination
            for i in range(min(3, len(memories)-1)):
                combo = {
                    'type': 'analogy',
                    'elements': [
                        {'id': memories[i].get('memory_id'), 
                         'content': memories[i].get('data')},
                        {'id': memories[i+1].get('memory_id'), 
                         'content': memories[i+1].get('data')}
                    ]
                }
                combinations.append(combo)
                
        if len(memories) >= 3:
            # Synthesis combination
            combo = {
                'type': 'synthesis',
                'elements': [
                    {'id': m.get('memory_id'), 'content': m.get('data')}
                    for m in memories[:3]
                ]
            }
            combinations.append(combo)
            
        return combinations
        
    def _construct_narrative(self, memories: List[Dict]) -> Optional[Dict]:
        """Construct narrative from memory sequence"""
        if len(memories) < 3:
            return None
            
        # Sort by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.get('timestamp', ''))
        
        # Build narrative structure
        narrative = {
            'content': {
                'beginning': self._summarize_memory(sorted_memories[0]),
                'middle': [self._summarize_memory(m) for m in sorted_memories[1:-1]],
                'end': self._summarize_memory(sorted_memories[-1])
            },
            'timeline': {
                'start': sorted_memories[0].get('timestamp'),
                'end': sorted_memories[-1].get('timestamp')
            },
            'theme': 'experience_consolidation'
        }
        
        return narrative
        
    async def _analyze_memory_patterns(self, nova_id: str, 
                                     memories: List[Dict]) -> List[Dict]:
        """Analyze patterns in memory formation and access"""
        patterns = []
        
        # Temporal patterns
        timestamps = [datetime.fromisoformat(m.get('timestamp', '')) 
                     for m in memories if m.get('timestamp')]
        
        if timestamps:
            # Find peak activity times
            hours = [t.hour for t in timestamps]
            hour_counts = {}
            for hour in hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
                
            peak_hour = max(hour_counts.items(), key=lambda x: x[1])
            
            patterns.append({
                'pattern_type': 'temporal_activity',
                'peak_hour': peak_hour[0],
                'activity_distribution': hour_counts
            })
            
        # Context patterns
        contexts = [m.get('context', 'general') for m in memories]
        context_counts = {}
        for context in contexts:
            context_counts[context] = context_counts.get(context, 0) + 1
            
        if context_counts:
            patterns.append({
                'pattern_type': 'context_distribution',
                'primary_context': max(context_counts.items(), key=lambda x: x[1])[0],
                'distribution': context_counts
            })
            
        # Importance patterns
        importances = [m.get('importance', 0.5) for m in memories]
        if importances:
            patterns.append({
                'pattern_type': 'importance_profile',
                'average': np.mean(importances),
                'std': np.std(importances),
                'trend': 'increasing' if importances[-10:] > importances[:10] else 'stable'
            })
            
        return patterns
        
    def _generate_predictions(self, patterns: List[Dict]) -> List[Dict]:
        """Generate predictions based on discovered patterns"""
        predictions = []
        
        for pattern in patterns:
            if pattern['pattern_type'] == 'temporal_activity':
                predictions.append({
                    'prediction_type': 'activity_forecast',
                    'next_peak': pattern['peak_hour'],
                    'confidence': 0.7,
                    'basis': 'temporal_pattern'
                })
                
            elif pattern['pattern_type'] == 'context_distribution':
                predictions.append({
                    'prediction_type': 'context_likelihood',
                    'likely_context': pattern['primary_context'],
                    'probability': pattern['distribution'][pattern['primary_context']] / 
                                 sum(pattern['distribution'].values()),
                    'basis': 'context_pattern'
                })
                
        return predictions
        
    def _suggest_optimizations(self, memories: List[Dict]) -> List[Dict]:
        """Suggest memory organization optimizations"""
        suggestions = []
        
        # Check for redundancy
        contents = [str(m.get('data', {})) for m in memories]
        unique_contents = set(contents)
        
        if len(contents) > len(unique_contents) * 1.5:
            suggestions.append({
                'type': 'reduce_redundancy',
                'reason': 'High duplicate content detected',
                'action': 'Implement deduplication in write pipeline'
            })
            
        # Check for low importance memories
        low_importance = [m for m in memories if m.get('importance', 0.5) < 0.3]
        
        if len(low_importance) > len(memories) * 0.5:
            suggestions.append({
                'type': 'adjust_importance_threshold',
                'reason': 'Many low-importance memories',
                'action': 'Increase filtering threshold to 0.3'
            })
            
        return suggestions
        
    async def manual_consolidation(self, nova_id: str, 
                                 phase: ConsolidationPhase = ConsolidationPhase.SLOW_WAVE,
                                 time_range: timedelta = timedelta(days=1)) -> Dict[str, Any]:
        """Manually trigger consolidation for specific phase"""
        logger.info(f"Manual consolidation triggered for {nova_id} - Phase: {phase.value}")
        
        # Find matching cycle
        cycle = next((c for c in self.cycles if c.phase == phase), self.cycles[0])
        
        # Run consolidation
        self.current_phase = phase
        await self._consolidate_cycle(nova_id, cycle)
        
        return {
            'phase': phase.value,
            'consolidated': self.consolidation_stats['total_consolidated'],
            'patterns': self.consolidation_stats['patterns_discovered'],
            'insights': self.consolidation_stats['creative_insights']
        }
        
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status"""
        return {
            'is_running': self.is_running,
            'current_phase': self.current_phase.value,
            'statistics': self.consolidation_stats,
            'cycles_config': [
                {
                    'phase': c.phase.value,
                    'duration': c.duration.total_seconds(),
                    'memory_types': [mt.value for mt in c.memory_types],
                    'consolidation_rate': c.consolidation_rate
                }
                for c in self.cycles
            ]
        }

# Example usage
async def test_consolidation_engine():
    """Test the consolidation engine"""
    
    # Initialize components
    memory_api = NovaMemoryAPI()
    await memory_api.initialize()
    
    db_pool = memory_api.db_pool
    
    # Create consolidation engine
    engine = MemoryConsolidationEngine(memory_api, db_pool)
    await engine.initialize()
    
    # Test manual consolidation
    result = await engine.manual_consolidation(
        'bloom',
        ConsolidationPhase.SLOW_WAVE,
        timedelta(days=1)
    )
    
    print("Manual consolidation result:", json.dumps(result, indent=2))
    
    # Start automatic consolidation
    await engine.start_automatic_consolidation('bloom')
    
    # Let it run for a bit
    await asyncio.sleep(10)
    
    # Get status
    status = engine.get_consolidation_status()
    print("Consolidation status:", json.dumps(status, indent=2))
    
    # Stop consolidation
    await engine.stop_automatic_consolidation()
    
    await memory_api.shutdown()

if __name__ == "__main__":
    asyncio.run(test_consolidation_engine())