#!/usr/bin/env python3
"""
Neural Semantic Memory Optimization
Fuses Echo's Neural Memory Network with Bloom's Semantic Layers
Part of the Revolutionary Memory Architecture Project
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import networkx as nx
from collections import defaultdict

@dataclass
class NeuralPathway:
    """Represents a neural pathway in the memory network"""
    source_concept: str
    target_concept: str
    strength: float
    activation_count: int
    last_activated: datetime
    pathway_type: str  # associative, hierarchical, causal, temporal

@dataclass
class SemanticNode:
    """Semantic memory node with neural properties"""
    concept_id: str
    concept_name: str
    semantic_layer: str  # conceptual, factual, linguistic, cultural
    embedding: Optional[np.ndarray]
    activation_level: float
    connections: List[str]
    metadata: Dict[str, Any]

class NeuralMemoryNetwork:
    """
    Echo's Neural Memory Network implementation
    Self-organizing topology with Hebbian learning
    """
    
    def __init__(self):
        self.network = nx.DiGraph()
        self.pathways = {}
        self.activation_history = defaultdict(list)
        self.learning_rate = 0.1
        self.decay_rate = 0.01
        
    async def find_optimal_paths(self, concept: str, max_paths: int = 5) -> List[List[str]]:
        """Find optimal neural pathways for a concept - OPTIMIZED"""
        if concept not in self.network:
            return []
            
        # OPTIMIZATION: Use BFS with early termination for large networks
        if len(self.network.nodes()) > 100:
            return await self._find_paths_optimized(concept, max_paths)
            
        # Get all connected nodes within 3 hops
        paths = []
        
        # OPTIMIZATION: Pre-filter candidates by direct connection strength
        candidates = list(self.network.successors(concept))
        candidates.sort(key=lambda x: self.network[concept][x].get('strength', 0), reverse=True)
        candidates = candidates[:min(20, len(candidates))]  # Limit search space
        
        for target in candidates:
            try:
                # Find shortest paths weighted by inverse strength
                path_generator = nx.all_shortest_paths(
                    self.network, 
                    source=concept, 
                    target=target,
                    weight='inverse_strength'
                )
                
                for path in path_generator:
                    if len(path) <= 4:  # Max 3 hops
                        paths.append(path)
                        
                    if len(paths) >= max_paths:
                        break
                        
            except nx.NetworkXNoPath:
                continue
                
            if len(paths) >= max_paths:
                break
                
    async def _find_paths_optimized(self, concept: str, max_paths: int) -> List[List[str]]:
        """Optimized pathfinding for large networks"""
        paths = []
        visited = set()
        queue = [(concept, [concept])]
        
        while queue and len(paths) < max_paths:
            current, path = queue.pop(0)
            
            if len(path) > 4:  # Max 3 hops
                continue
                
            if current in visited and len(path) > 2:
                continue
                
            visited.add(current)
            
            # Get top 5 strongest connections only
            neighbors = [(n, self.network[current][n].get('strength', 0)) 
                        for n in self.network.successors(current)]
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            for neighbor, strength in neighbors[:5]:
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    if len(new_path) > 2:  # Valid path
                        paths.append(new_path)
                        if len(paths) >= max_paths:
                            break
                    queue.append((neighbor, new_path))
                    
        return paths[:max_paths]
                
        # Sort by total pathway strength
        scored_paths = []
        for path in paths:
            total_strength = self._calculate_path_strength(path)
            scored_paths.append((total_strength, path))
            
        scored_paths.sort(reverse=True, key=lambda x: x[0])
        
        return [path for _, path in scored_paths[:max_paths]]
        
    def _calculate_path_strength(self, path: List[str]) -> float:
        """Calculate total strength of a pathway"""
        if len(path) < 2:
            return 0.0
            
        total_strength = 0.0
        for i in range(len(path) - 1):
            edge_data = self.network.get_edge_data(path[i], path[i+1])
            if edge_data:
                total_strength += edge_data.get('strength', 0.0)
                
        return total_strength / (len(path) - 1)
        
    async def strengthen_pathways(self, paths: List[List[str]], reward: float = 1.0):
        """Hebbian learning - strengthen successful pathways"""
        for path in paths:
            for i in range(len(path) - 1):
                source, target = path[i], path[i+1]
                
                # Update edge strength
                if self.network.has_edge(source, target):
                    current_strength = self.network[source][target]['strength']
                    new_strength = current_strength + self.learning_rate * reward
                    new_strength = min(1.0, new_strength)  # Cap at 1.0
                    
                    self.network[source][target]['strength'] = new_strength
                    self.network[source][target]['activation_count'] += 1
                    self.network[source][target]['last_activated'] = datetime.now()
                    
                    # Update inverse for pathfinding
                    self.network[source][target]['inverse_strength'] = 1.0 / new_strength
                    
        # Apply decay to unused pathways
        await self._apply_decay()
        
    async def _apply_decay(self):
        """Apply decay to unused pathways"""
        current_time = datetime.now()
        
        for source, target, data in self.network.edges(data=True):
            last_activated = data.get('last_activated', current_time)
            time_diff = (current_time - last_activated).total_seconds() / 3600  # Hours
            
            if time_diff > 24:  # No activation in 24 hours
                decay_factor = self.decay_rate * (time_diff / 24)
                new_strength = data['strength'] * (1 - decay_factor)
                new_strength = max(0.01, new_strength)  # Minimum strength
                
                self.network[source][target]['strength'] = new_strength
                self.network[source][target]['inverse_strength'] = 1.0 / new_strength
                
    def add_neural_connection(self, source: str, target: str, 
                            initial_strength: float = 0.1):
        """Add a new neural connection"""
        self.network.add_edge(
            source, target,
            strength=initial_strength,
            inverse_strength=1.0 / initial_strength,
            activation_count=0,
            last_activated=datetime.now(),
            pathway_type='associative'
        )
        
class BloomSemanticLayers:
    """
    Bloom's Semantic Memory Layers
    Enhanced with neural network optimization
    """
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.layers = {
            'conceptual': {
                'description': 'Abstract concepts and ideas',
                'examples': ['justice', 'beauty', 'consciousness']
            },
            'factual': {
                'description': 'Concrete facts and information',
                'examples': ['Earth orbits Sun', 'Water boils at 100C']
            },
            'linguistic': {
                'description': 'Language patterns and structures',
                'examples': ['grammar rules', 'vocabulary', 'idioms']
            },
            'cultural': {
                'description': 'Cultural knowledge and norms',
                'examples': ['traditions', 'social rules', 'customs']
            },
            'procedural_semantic': {
                'description': 'How-to knowledge representations',
                'examples': ['cooking methods', 'problem-solving strategies']
            },
            'relational': {
                'description': 'Relationships between concepts',
                'examples': ['is-a', 'part-of', 'causes', 'related-to']
            }
        }
        
    async def traverse(self, pathway: List[str], layers: List[str]) -> Dict[str, Any]:
        """Traverse semantic layers along a neural pathway"""
        knowledge_graph = {}
        
        for node in pathway:
            node_knowledge = {}
            
            for layer in layers:
                if layer not in self.layers:
                    continue
                    
                # Query layer for this concept
                layer_knowledge = await self._query_semantic_layer(node, layer)
                if layer_knowledge:
                    node_knowledge[layer] = layer_knowledge
                    
            if node_knowledge:
                knowledge_graph[node] = node_knowledge
                
        return knowledge_graph
        
    async def _query_semantic_layer(self, concept: str, layer: str) -> Optional[Dict[str, Any]]:
        """Query specific semantic layer for a concept"""
        dragonfly = self.db_pool.get_connection('dragonfly')
        
        key = f"nova:semantic:{layer}:{concept}"
        data = dragonfly.get(key)
        
        if data:
            return json.loads(data)
            
        # Try pattern matching
        pattern = f"nova:semantic:{layer}:*{concept}*"
        cursor = 0
        matches = []
        
        while True:
            cursor, keys = dragonfly.scan(cursor, match=pattern, count=10)
            
            for key in keys[:3]:  # Limit to 3 matches
                match_data = dragonfly.get(key)
                if match_data:
                    matches.append(json.loads(match_data))
                    
            if cursor == 0 or len(matches) >= 3:
                break
                
        return {'matches': matches} if matches else None
        
    async def store_semantic_knowledge(self, node: SemanticNode):
        """Store semantic knowledge in appropriate layer"""
        dragonfly = self.db_pool.get_connection('dragonfly')
        
        key = f"nova:semantic:{node.semantic_layer}:{node.concept_id}"
        
        data = {
            'concept_id': node.concept_id,
            'concept_name': node.concept_name,
            'layer': node.semantic_layer,
            'activation_level': node.activation_level,
            'connections': node.connections,
            'metadata': node.metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store with vector embedding if available
        if node.embedding is not None:
            data['embedding'] = node.embedding.tolist()
            
        dragonfly.set(key, json.dumps(data))
        
        # Update connections index
        for connection in node.connections:
            dragonfly.sadd(f"nova:semantic:connections:{connection}", node.concept_id)
            
class NeuralSemanticMemory:
    """
    Unified Neural-Semantic Memory System
    Combines Echo's neural pathways with Bloom's semantic layers
    """
    
    def __init__(self, db_pool):
        self.neural_network = NeuralMemoryNetwork()
        self.semantic_layers = BloomSemanticLayers(db_pool)
        self.concept_embeddings = {}
        self.activation_threshold = 0.3
        
    async def optimize_semantic_access(self, query_concept: str, 
                                     target_layers: List[str] = None) -> Dict[str, Any]:
        """
        Optimize semantic memory access using neural pathways
        """
        if target_layers is None:
            target_layers = ['conceptual', 'factual', 'relational']
            
        # Find optimal neural pathways
        pathways = await self.neural_network.find_optimal_paths(query_concept)
        
        if not pathways:
            # Create new pathway if none exists
            await self._explore_new_pathways(query_concept)
            pathways = await self.neural_network.find_optimal_paths(query_concept)
            
        # Traverse semantic layers along pathways
        semantic_results = []
        pathway_knowledge = {}
        
        for pathway in pathways:
            knowledge = await self.semantic_layers.traverse(pathway, target_layers)
            
            if knowledge:
                semantic_results.append({
                    'pathway': pathway,
                    'knowledge': knowledge,
                    'strength': self.neural_network._calculate_path_strength(pathway)
                })
                
                # Merge knowledge
                for concept, layers in knowledge.items():
                    if concept not in pathway_knowledge:
                        pathway_knowledge[concept] = {}
                    pathway_knowledge[concept].update(layers)
                    
        # Strengthen successful pathways
        if semantic_results:
            successful_paths = [r['pathway'] for r in semantic_results]
            await self.neural_network.strengthen_pathways(successful_paths)
            
        return {
            'query_concept': query_concept,
            'pathways_found': len(pathways),
            'semantic_results': semantic_results,
            'unified_knowledge': pathway_knowledge,
            'network_updated': True
        }
        
    async def _explore_new_pathways(self, concept: str):
        """Explore and create new neural pathways"""
        # Look for related concepts in semantic layers
        dragonfly = self.semantic_layers.db_pool.get_connection('dragonfly')
        
        # Find concepts that share connections
        related_concepts = set()
        
        # Search across all layers
        for layer in self.semantic_layers.layers:
            pattern = f"nova:semantic:{layer}:*"
            cursor = 0
            
            while True:
                cursor, keys = dragonfly.scan(cursor, match=pattern, count=100)
                
                for key in keys:
                    data = dragonfly.get(key)
                    if data:
                        node_data = json.loads(data)
                        
                        # Check if this concept is related
                        if concept in str(node_data).lower():
                            concept_id = node_data.get('concept_id', key.split(':')[-1])
                            related_concepts.add(concept_id)
                            
                if cursor == 0:
                    break
                    
        # Create neural connections to related concepts
        for related in related_concepts:
            if related != concept:
                self.neural_network.add_neural_connection(concept, related, 0.2)
                
        # Also add bidirectional connections for strong relationships
        for related in list(related_concepts)[:5]:  # Top 5
            self.neural_network.add_neural_connection(related, concept, 0.15)
            
    async def create_semantic_association(self, concept_a: str, concept_b: str, 
                                        association_type: str, strength: float = 0.5):
        """Create a semantic association with neural pathway"""
        # Add neural connection
        self.neural_network.add_neural_connection(concept_a, concept_b, strength)
        
        # Store semantic relationship
        dragonfly = self.semantic_layers.db_pool.get_connection('dragonfly')
        
        association_data = {
            'source': concept_a,
            'target': concept_b,
            'type': association_type,
            'strength': strength,
            'created': datetime.now().isoformat()
        }
        
        # Store bidirectionally
        dragonfly.sadd(f"nova:semantic:associations:{concept_a}", json.dumps(association_data))
        
        # Reverse association
        reverse_data = association_data.copy()
        reverse_data['source'] = concept_b
        reverse_data['target'] = concept_a
        dragonfly.sadd(f"nova:semantic:associations:{concept_b}", json.dumps(reverse_data))
        
    async def propagate_activation(self, initial_concept: str, 
                                 activation_energy: float = 1.0) -> Dict[str, float]:
        """Propagate activation through neural-semantic network"""
        activation_levels = {initial_concept: activation_energy}
        to_process = [(initial_concept, activation_energy)]
        processed = set()
        
        while to_process:
            current_concept, current_energy = to_process.pop(0)
            
            if current_concept in processed:
                continue
                
            processed.add(current_concept)
            
            # Get neural connections
            if current_concept in self.neural_network.network:
                neighbors = self.neural_network.network.neighbors(current_concept)
                
                for neighbor in neighbors:
                    edge_data = self.neural_network.network[current_concept][neighbor]
                    strength = edge_data['strength']
                    
                    # Calculate propagated activation
                    propagated_energy = current_energy * strength * 0.7  # Decay factor
                    
                    if propagated_energy > self.activation_threshold:
                        if neighbor not in activation_levels:
                            activation_levels[neighbor] = 0
                            
                        activation_levels[neighbor] += propagated_energy
                        
                        if neighbor not in processed:
                            to_process.append((neighbor, propagated_energy))
                            
        return activation_levels
        
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get neural network statistics"""
        return {
            'total_nodes': self.neural_network.network.number_of_nodes(),
            'total_connections': self.neural_network.network.number_of_edges(),
            'average_degree': np.mean([d for n, d in self.neural_network.network.degree()]) if self.neural_network.network.number_of_nodes() > 0 else 0,
            'strongly_connected_components': nx.number_strongly_connected_components(self.neural_network.network),
            'network_density': nx.density(self.neural_network.network)
        }

# Example usage
async def demonstrate_neural_semantic():
    """Demonstrate neural semantic memory capabilities"""
    from database_connections import NovaDatabasePool
    
    # Initialize database pool
    db_pool = NovaDatabasePool()
    await db_pool.initialize_all_connections()
    
    # Create neural semantic memory system
    nsm = NeuralSemanticMemory(db_pool)
    
    # Store some semantic knowledge
    concepts = [
        SemanticNode(
            concept_id="consciousness",
            concept_name="Consciousness",
            semantic_layer="conceptual",
            embedding=np.random.randn(768),  # Simulated embedding
            activation_level=0.9,
            connections=["awareness", "mind", "experience", "qualia"],
            metadata={"definition": "The state of being aware of and able to think"}
        ),
        SemanticNode(
            concept_id="memory",
            concept_name="Memory",
            semantic_layer="conceptual",
            embedding=np.random.randn(768),
            activation_level=0.8,
            connections=["consciousness", "storage", "recall", "experience"],
            metadata={"definition": "The faculty by which information is encoded, stored, and retrieved"}
        )
    ]
    
    # Store concepts
    for concept in concepts:
        await nsm.semantic_layers.store_semantic_knowledge(concept)
        
    # Create neural pathways
    nsm.neural_network.add_neural_connection("consciousness", "memory", 0.9)
    nsm.neural_network.add_neural_connection("memory", "experience", 0.8)
    nsm.neural_network.add_neural_connection("experience", "qualia", 0.7)
    
    # Optimize semantic access
    print("🧠 Optimizing semantic access for 'consciousness'...")
    results = await nsm.optimize_semantic_access("consciousness")
    
    print(f"✅ Found {results['pathways_found']} neural pathways")
    print(f"📊 Network statistics: {nsm.get_network_statistics()}")
    
    # Test activation propagation
    print("\n⚡ Testing activation propagation...")
    activation = await nsm.propagate_activation("consciousness", 1.0)
    print(f"🌊 Activation spread to {len(activation)} concepts")
    
    for concept, level in sorted(activation.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {concept}: {level:.3f}")

if __name__ == "__main__":
    asyncio.run(demonstrate_neural_semantic())