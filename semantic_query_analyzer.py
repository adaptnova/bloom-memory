#!/usr/bin/env python3
"""
Nova Memory System - Semantic Query Analyzer
Advanced NLP-powered query understanding and semantic optimization
"""

import json
import re
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import math

logger = logging.getLogger(__name__)

class SemanticIntent(Enum):
    """Semantic intent classification"""
    RETRIEVE_MEMORY = "retrieve_memory"
    STORE_MEMORY = "store_memory"
    UPDATE_MEMORY = "update_memory"
    ANALYZE_MEMORY = "analyze_memory"
    SEARCH_SIMILARITY = "search_similarity"
    TEMPORAL_QUERY = "temporal_query"
    CONTEXTUAL_QUERY = "contextual_query"
    RELATIONSHIP_QUERY = "relationship_query"
    PATTERN_QUERY = "pattern_query"
    SUMMARIZATION = "summarization"

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4

class MemoryDomain(Enum):
    """Memory domain classifications"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    SENSORY = "sensory"
    METACOGNITIVE = "metacognitive"
    CREATIVE = "creative"
    LINGUISTIC = "linguistic"

@dataclass
class SemanticEntity:
    """Semantic entity extracted from query"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticRelation:
    """Semantic relationship between entities"""
    subject: SemanticEntity
    predicate: str
    object: SemanticEntity
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QuerySemantics:
    """Comprehensive semantic analysis of query"""
    original_query: Dict[str, Any]
    intent: SemanticIntent
    complexity: QueryComplexity
    domains: List[MemoryDomain]
    entities: List[SemanticEntity]
    relations: List[SemanticRelation]
    temporal_aspects: Dict[str, Any]
    spatial_aspects: Dict[str, Any]
    emotional_aspects: Dict[str, Any]
    confidence_score: float
    suggested_rewrites: List[Dict[str, Any]]
    optimization_hints: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SemanticPattern:
    """Semantic pattern in queries"""
    pattern_id: str
    pattern_type: str
    pattern_description: str
    frequency: int
    examples: List[str]
    optimization_benefit: float
    last_seen: datetime = field(default_factory=datetime.utcnow)

class SemanticVocabulary:
    """Vocabulary for semantic understanding"""
    
    # Intent keywords mapping
    INTENT_KEYWORDS = {
        SemanticIntent.RETRIEVE_MEMORY: [
            'get', 'find', 'retrieve', 'recall', 'remember', 'lookup', 'fetch',
            'search', 'query', 'show', 'display', 'list'
        ],
        SemanticIntent.STORE_MEMORY: [
            'store', 'save', 'remember', 'record', 'memorize', 'keep', 'retain',
            'preserve', 'archive', 'log', 'write', 'create'
        ],
        SemanticIntent.UPDATE_MEMORY: [
            'update', 'modify', 'change', 'edit', 'revise', 'alter', 'correct',
            'amend', 'adjust', 'refine'
        ],
        SemanticIntent.ANALYZE_MEMORY: [
            'analyze', 'examine', 'study', 'investigate', 'explore', 'review',
            'assess', 'evaluate', 'inspect', 'scrutinize'
        ],
        SemanticIntent.SEARCH_SIMILARITY: [
            'similar', 'like', 'related', 'comparable', 'analogous', 'resembling',
            'matching', 'parallel', 'corresponding'
        ],
        SemanticIntent.TEMPORAL_QUERY: [
            'when', 'before', 'after', 'during', 'since', 'until', 'recent',
            'past', 'future', 'yesterday', 'today', 'tomorrow', 'ago'
        ],
        SemanticIntent.CONTEXTUAL_QUERY: [
            'context', 'situation', 'circumstance', 'environment', 'setting',
            'background', 'condition', 'scenario'
        ],
        SemanticIntent.RELATIONSHIP_QUERY: [
            'relationship', 'connection', 'association', 'link', 'correlation',
            'causation', 'influence', 'dependency', 'interaction'
        ],
        SemanticIntent.PATTERN_QUERY: [
            'pattern', 'trend', 'sequence', 'cycle', 'routine', 'habit',
            'recurring', 'repeated', 'regular'
        ],
        SemanticIntent.SUMMARIZATION: [
            'summary', 'summarize', 'overview', 'gist', 'essence', 'synopsis',
            'abstract', 'condensed', 'brief'
        ]
    }
    
    # Domain keywords mapping
    DOMAIN_KEYWORDS = {
        MemoryDomain.EPISODIC: [
            'experience', 'event', 'episode', 'moment', 'incident', 'occurrence',
            'happening', 'story', 'narrative', 'autobiography'
        ],
        MemoryDomain.SEMANTIC: [
            'knowledge', 'fact', 'concept', 'meaning', 'definition', 'understanding',
            'information', 'data', 'wisdom', 'insight'
        ],
        MemoryDomain.PROCEDURAL: [
            'procedure', 'process', 'method', 'technique', 'skill', 'ability',
            'know-how', 'practice', 'routine', 'workflow'
        ],
        MemoryDomain.WORKING: [
            'current', 'active', 'immediate', 'present', 'ongoing', 'temporary',
            'short-term', 'buffer', 'cache'
        ],
        MemoryDomain.EMOTIONAL: [
            'emotion', 'feeling', 'mood', 'sentiment', 'affect', 'emotional',
            'happy', 'sad', 'angry', 'fear', 'joy', 'love', 'hate'
        ],
        MemoryDomain.SOCIAL: [
            'social', 'people', 'person', 'relationship', 'interaction', 'communication',
            'friend', 'family', 'colleague', 'community', 'group'
        ],
        MemoryDomain.SENSORY: [
            'sensory', 'visual', 'auditory', 'tactile', 'smell', 'taste',
            'see', 'hear', 'feel', 'touch', 'sound', 'image'
        ],
        MemoryDomain.METACOGNITIVE: [
            'thinking', 'cognition', 'awareness', 'consciousness', 'reflection',
            'introspection', 'self-awareness', 'mindfulness'
        ],
        MemoryDomain.CREATIVE: [
            'creative', 'imagination', 'idea', 'innovation', 'inspiration',
            'artistic', 'original', 'novel', 'inventive'
        ],
        MemoryDomain.LINGUISTIC: [
            'language', 'word', 'text', 'speech', 'communication', 'verbal',
            'linguistic', 'sentence', 'phrase', 'vocabulary'
        ]
    }
    
    # Temporal keywords
    TEMPORAL_KEYWORDS = {
        'absolute_time': ['date', 'time', 'timestamp', 'when', 'at'],
        'relative_time': ['before', 'after', 'during', 'since', 'until', 'ago'],
        'frequency': ['daily', 'weekly', 'monthly', 'often', 'rarely', 'sometimes'],
        'duration': ['for', 'throughout', 'lasting', 'span', 'period']
    }
    
    # Spatial keywords
    SPATIAL_KEYWORDS = {
        'location': ['where', 'place', 'location', 'position', 'site'],
        'direction': ['north', 'south', 'east', 'west', 'up', 'down', 'left', 'right'],
        'proximity': ['near', 'far', 'close', 'distant', 'adjacent', 'nearby'],
        'containment': ['in', 'inside', 'within', 'outside', 'around']
    }
    
    # Emotional keywords
    EMOTIONAL_KEYWORDS = {
        'positive': ['happy', 'joy', 'excited', 'pleased', 'satisfied', 'content'],
        'negative': ['sad', 'angry', 'frustrated', 'disappointed', 'worried', 'anxious'],
        'intensity': ['very', 'extremely', 'highly', 'moderately', 'slightly', 'somewhat']
    }

class SemanticQueryAnalyzer:
    """
    Advanced semantic analyzer for Nova memory queries
    Provides NLP-powered query understanding and optimization
    """
    
    def __init__(self):
        self.vocabulary = SemanticVocabulary()
        self.pattern_cache = {}
        self.analysis_cache = {}
        self.semantic_patterns = []
        
        # Statistics
        self.analysis_stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'intent_distribution': defaultdict(int),
            'domain_distribution': defaultdict(int),
            'complexity_distribution': defaultdict(int)
        }
        
        logger.info("Semantic Query Analyzer initialized")
    
    async def analyze_query(self, query: Dict[str, Any], 
                          context: Optional[Dict[str, Any]] = None) -> QuerySemantics:
        """
        Main semantic analysis entry point
        Returns comprehensive semantic understanding of query
        """
        self.analysis_stats['total_analyses'] += 1
        
        # Check cache first
        query_hash = self._generate_query_hash(query)
        if query_hash in self.analysis_cache:
            self.analysis_stats['cache_hits'] += 1
            return self.analysis_cache[query_hash]
        
        # Extract text content from query
        query_text = self._extract_query_text(query)
        
        # Perform semantic analysis
        semantics = await self._perform_semantic_analysis(query, query_text, context)
        
        # Cache the result
        self.analysis_cache[query_hash] = semantics
        
        # Update statistics
        self.analysis_stats['intent_distribution'][semantics.intent.value] += 1
        self.analysis_stats['complexity_distribution'][semantics.complexity.value] += 1
        for domain in semantics.domains:
            self.analysis_stats['domain_distribution'][domain.value] += 1
        
        # Update semantic patterns
        await self._update_semantic_patterns(semantics)
        
        logger.debug(f"Query analyzed - Intent: {semantics.intent.value}, "
                    f"Complexity: {semantics.complexity.value}, "
                    f"Domains: {[d.value for d in semantics.domains]}")
        
        return semantics
    
    async def suggest_query_optimizations(self, semantics: QuerySemantics) -> List[Dict[str, Any]]:
        """Generate query optimization suggestions based on semantic analysis"""
        optimizations = []
        
        # Intent-based optimizations
        if semantics.intent == SemanticIntent.SEARCH_SIMILARITY:
            optimizations.append({
                'type': 'indexing',
                'suggestion': 'Use vector similarity indexes for semantic search',
                'benefit': 'Significant performance improvement for similarity queries',
                'implementation': 'Create vector embeddings and similarity index'
            })
        
        elif semantics.intent == SemanticIntent.TEMPORAL_QUERY:
            optimizations.append({
                'type': 'temporal_indexing',
                'suggestion': 'Use temporal indexes for time-based queries',
                'benefit': 'Faster temporal range queries and sorting',
                'implementation': 'Create B-tree index on timestamp columns'
            })
        
        # Domain-based optimizations
        if MemoryDomain.EPISODIC in semantics.domains:
            optimizations.append({
                'type': 'partitioning',
                'suggestion': 'Partition episodic data by time periods',
                'benefit': 'Improved query performance for recent memories',
                'implementation': 'Implement time-based partitioning strategy'
            })
        
        # Complexity-based optimizations
        if semantics.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            optimizations.append({
                'type': 'query_decomposition',
                'suggestion': 'Break complex query into simpler sub-queries',
                'benefit': 'Better parallelization and resource utilization',
                'implementation': 'Implement query decomposition strategy'
            })
        
        # Entity-based optimizations
        if len(semantics.entities) > 3:
            optimizations.append({
                'type': 'entity_preprocessing',
                'suggestion': 'Pre-process entities for faster matching',
                'benefit': 'Reduced entity resolution overhead',
                'implementation': 'Create entity lookup cache'
            })
        
        return optimizations
    
    async def rewrite_query_for_optimization(self, semantics: QuerySemantics) -> List[Dict[str, Any]]:
        """Generate semantically equivalent but optimized query rewrites"""
        rewrites = []
        
        original_query = semantics.original_query
        
        # Simplification rewrites
        if semantics.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            # Break into sub-queries
            sub_queries = await self._decompose_complex_query(semantics)
            if sub_queries:
                rewrites.append({
                    'type': 'decomposition',
                    'original': original_query,
                    'rewritten': sub_queries,
                    'benefit': 'Improved parallelization and caching',
                    'confidence': 0.8
                })
        
        # Index-aware rewrites
        if semantics.intent == SemanticIntent.SEARCH_SIMILARITY:
            # Suggest vector search rewrite
            vector_query = await self._rewrite_for_vector_search(semantics)
            if vector_query:
                rewrites.append({
                    'type': 'vector_search',
                    'original': original_query,
                    'rewritten': vector_query,
                    'benefit': 'Leverages semantic similarity indexes',
                    'confidence': 0.9
                })
        
        # Temporal optimization rewrites
        if semantics.temporal_aspects:
            temporal_query = await self._rewrite_for_temporal_optimization(semantics)
            if temporal_query:
                rewrites.append({
                    'type': 'temporal_optimization',
                    'original': original_query,
                    'rewritten': temporal_query,
                    'benefit': 'Optimized temporal range queries',
                    'confidence': 0.85
                })
        
        # Filter pushdown rewrites
        if len(semantics.entities) > 0:
            filter_optimized = await self._rewrite_for_filter_pushdown(semantics)
            if filter_optimized:
                rewrites.append({
                    'type': 'filter_pushdown',
                    'original': original_query,
                    'rewritten': filter_optimized,
                    'benefit': 'Reduces data processing volume',
                    'confidence': 0.7
                })
        
        return rewrites
    
    async def detect_query_patterns(self, query_history: List[QuerySemantics], 
                                  time_window_hours: int = 24) -> List[SemanticPattern]:
        """Detect recurring semantic patterns in query history"""
        if not query_history:
            return []
        
        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_queries = [q for q in query_history if q.created_at > cutoff_time]
        
        patterns = []
        
        # Intent patterns
        intent_counts = Counter([q.intent for q in recent_queries])
        for intent, count in intent_counts.most_common(5):
            if count >= 3:  # Pattern threshold
                pattern = SemanticPattern(
                    pattern_id=f"intent_{intent.value}",
                    pattern_type="intent_frequency",
                    pattern_description=f"Frequent {intent.value} queries",
                    frequency=count,
                    examples=[str(q.original_query)[:100] for q in recent_queries 
                             if q.intent == intent][:3],
                    optimization_benefit=self._calculate_pattern_benefit(intent, count)
                )
                patterns.append(pattern)
        
        # Domain patterns
        domain_combinations = []
        for q in recent_queries:
            domain_set = tuple(sorted([d.value for d in q.domains]))
            domain_combinations.append(domain_set)
        
        domain_counts = Counter(domain_combinations)
        for domains, count in domain_counts.most_common(3):
            if count >= 2:
                pattern = SemanticPattern(
                    pattern_id=f"domains_{'_'.join(domains)}",
                    pattern_type="domain_combination",
                    pattern_description=f"Queries spanning domains: {', '.join(domains)}",
                    frequency=count,
                    examples=[str(q.original_query)[:100] for q in recent_queries 
                             if tuple(sorted([d.value for d in q.domains])) == domains][:2],
                    optimization_benefit=count * 0.2  # Base benefit
                )
                patterns.append(pattern)
        
        # Entity patterns
        entity_types = []
        for q in recent_queries:
            for entity in q.entities:
                entity_types.append(entity.entity_type)
        
        entity_counts = Counter(entity_types)
        for entity_type, count in entity_counts.most_common(3):
            if count >= 3:
                pattern = SemanticPattern(
                    pattern_id=f"entity_{entity_type}",
                    pattern_type="entity_frequency",
                    pattern_description=f"Frequent queries with {entity_type} entities",
                    frequency=count,
                    examples=[],  # Would extract relevant examples
                    optimization_benefit=count * 0.15
                )
                patterns.append(pattern)
        
        # Update pattern cache
        self.semantic_patterns.extend(patterns)
        self.semantic_patterns = self.semantic_patterns[-1000:]  # Keep recent patterns
        
        return patterns
    
    def get_semantic_statistics(self) -> Dict[str, Any]:
        """Get comprehensive semantic analysis statistics"""
        return {
            'analysis_stats': dict(self.analysis_stats),
            'cache_size': len(self.analysis_cache),
            'pattern_count': len(self.semantic_patterns),
            'vocabulary_size': {
                'intent_keywords': sum(len(keywords) for keywords in 
                                     self.vocabulary.INTENT_KEYWORDS.values()),
                'domain_keywords': sum(len(keywords) for keywords in
                                     self.vocabulary.DOMAIN_KEYWORDS.values())
            }
        }
    
    def _generate_query_hash(self, query: Dict[str, Any]) -> str:
        """Generate hash for query caching"""
        return hashlib.md5(json.dumps(query, sort_keys=True).encode()).hexdigest()
    
    def _extract_query_text(self, query: Dict[str, Any]) -> str:
        """Extract text content from structured query"""
        text_parts = []
        
        # Extract from common query fields
        for field in ['query', 'search', 'text', 'description', 'content', 'summary']:
            if field in query and isinstance(query[field], str):
                text_parts.append(query[field])
        
        # Extract from conditions
        if 'conditions' in query:
            conditions = query['conditions']
            if isinstance(conditions, dict):
                for key, value in conditions.items():
                    if isinstance(value, str):
                        text_parts.append(f"{key} {value}")
            elif isinstance(conditions, str):
                text_parts.append(conditions)
        
        # Extract from filters
        if 'filters' in query:
            filters = query['filters']
            if isinstance(filters, list):
                for f in filters:
                    if isinstance(f, str):
                        text_parts.append(f)
                    elif isinstance(f, dict) and 'value' in f:
                        text_parts.append(str(f['value']))
        
        return ' '.join(text_parts).strip()
    
    async def _perform_semantic_analysis(self, query: Dict[str, Any], 
                                       query_text: str,
                                       context: Optional[Dict[str, Any]]) -> QuerySemantics:
        """Perform comprehensive semantic analysis"""
        
        # Classify intent
        intent = self._classify_intent(query, query_text)
        
        # Determine complexity
        complexity = self._calculate_complexity(query, query_text)
        
        # Identify domains
        domains = self._identify_domains(query, query_text)
        
        # Extract entities
        entities = self._extract_entities(query_text)
        
        # Extract relations
        relations = self._extract_relations(entities, query_text)
        
        # Analyze temporal aspects
        temporal_aspects = self._analyze_temporal_aspects(query, query_text)
        
        # Analyze spatial aspects
        spatial_aspects = self._analyze_spatial_aspects(query, query_text)
        
        # Analyze emotional aspects
        emotional_aspects = self._analyze_emotional_aspects(query, query_text)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            intent, complexity, domains, entities, relations
        )
        
        # Generate optimization hints
        optimization_hints = self._generate_optimization_hints(
            intent, complexity, domains, entities, temporal_aspects
        )
        
        return QuerySemantics(
            original_query=query,
            intent=intent,
            complexity=complexity,
            domains=domains,
            entities=entities,
            relations=relations,
            temporal_aspects=temporal_aspects,
            spatial_aspects=spatial_aspects,
            emotional_aspects=emotional_aspects,
            confidence_score=confidence_score,
            suggested_rewrites=[],  # Will be populated by rewrite methods
            optimization_hints=optimization_hints
        )
    
    def _classify_intent(self, query: Dict[str, Any], query_text: str) -> SemanticIntent:
        """Classify the semantic intent of the query"""
        text_lower = query_text.lower()
        intent_scores = {}
        
        # Check for explicit operation
        if 'operation' in query:
            operation = query['operation'].lower()
            if operation in ['read', 'get', 'find', 'search']:
                return SemanticIntent.RETRIEVE_MEMORY
            elif operation in ['write', 'insert', 'create', 'store']:
                return SemanticIntent.STORE_MEMORY
            elif operation in ['update', 'modify', 'edit']:
                return SemanticIntent.UPDATE_MEMORY
            elif operation in ['analyze', 'examine']:
                return SemanticIntent.ANALYZE_MEMORY
        
        # Score based on keywords
        for intent, keywords in self.vocabulary.INTENT_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword importance and frequency
                    frequency = text_lower.count(keyword)
                    score += frequency * (1.0 / len(keyword))  # Shorter words get higher weight
            intent_scores[intent] = score
        
        # Return highest scoring intent or default
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        return SemanticIntent.RETRIEVE_MEMORY  # Default
    
    def _calculate_complexity(self, query: Dict[str, Any], query_text: str) -> QueryComplexity:
        """Calculate query complexity based on various factors"""
        complexity_score = 0
        
        # Text length factor
        word_count = len(query_text.split())
        if word_count > 50:
            complexity_score += 3
        elif word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Nested structure factor
        def count_nested_dicts(obj, depth=0):
            if isinstance(obj, dict):
                max_depth = depth
                for value in obj.values():
                    child_depth = count_nested_dicts(value, depth + 1)
                    max_depth = max(max_depth, child_depth)
                return max_depth
            elif isinstance(obj, list):
                max_depth = depth
                for item in obj:
                    child_depth = count_nested_dicts(item, depth)
                    max_depth = max(max_depth, child_depth)
                return max_depth
            return depth
        
        nesting_depth = count_nested_dicts(query)
        if nesting_depth > 4:
            complexity_score += 3
        elif nesting_depth > 2:
            complexity_score += 2
        elif nesting_depth > 1:
            complexity_score += 1
        
        # Multiple conditions factor
        conditions_count = 0
        if 'conditions' in query:
            if isinstance(query['conditions'], list):
                conditions_count = len(query['conditions'])
            elif isinstance(query['conditions'], dict):
                conditions_count = len(query['conditions'])
        
        if conditions_count > 5:
            complexity_score += 2
        elif conditions_count > 2:
            complexity_score += 1
        
        # Joins and relationships
        if any(key in query for key in ['joins', 'relationships', 'associations']):
            complexity_score += 2
        
        # Aggregations
        if any(key in query for key in ['group_by', 'aggregation', 'sum', 'count', 'avg']):
            complexity_score += 1
        
        # Subqueries
        if 'subquery' in str(query) or 'subqueries' in query:
            complexity_score += 2
        
        # Map to complexity enum
        if complexity_score >= 8:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 5:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _identify_domains(self, query: Dict[str, Any], query_text: str) -> List[MemoryDomain]:
        """Identify relevant memory domains"""
        text_lower = query_text.lower()
        domain_scores = {}
        
        # Score domains based on keywords
        for domain, keywords in self.vocabulary.DOMAIN_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    frequency = text_lower.count(keyword)
                    score += frequency * (1.0 / len(keyword))
            if score > 0:
                domain_scores[domain] = score
        
        # Check explicit domain specification
        if 'memory_types' in query:
            memory_types = query['memory_types']
            if isinstance(memory_types, list):
                for mem_type in memory_types:
                    for domain in MemoryDomain:
                        if domain.value in mem_type.lower():
                            domain_scores[domain] = domain_scores.get(domain, 0) + 2.0
        
        # Check scope
        if 'scope' in query:
            scope = query['scope'].lower()
            for domain in MemoryDomain:
                if domain.value in scope:
                    domain_scores[domain] = domain_scores.get(domain, 0) + 1.5
        
        # Return top scoring domains (threshold = 0.5)
        relevant_domains = [
            domain for domain, score in domain_scores.items() 
            if score >= 0.5
        ]
        
        # Sort by score
        relevant_domains.sort(key=lambda d: domain_scores[d], reverse=True)
        
        # Default to working memory if no domains identified
        if not relevant_domains:
            relevant_domains = [MemoryDomain.WORKING]
        
        return relevant_domains[:5]  # Limit to top 5 domains
    
    def _extract_entities(self, query_text: str) -> List[SemanticEntity]:
        """Extract semantic entities from query text"""
        entities = []
        
        # Simple entity extraction (in production, use NER models)
        # Extract dates
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{4}\b',  # MM-DD-YYYY
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, query_text):
                entities.append(SemanticEntity(
                    text=match.group(),
                    entity_type='date',
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        # Extract times
        time_pattern = r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'
        for match in re.finditer(time_pattern, query_text):
            entities.append(SemanticEntity(
                text=match.group(),
                entity_type='time',
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        # Extract quoted strings (likely important terms)
        quote_pattern = r'"([^"]+)"'
        for match in re.finditer(quote_pattern, query_text):
            entities.append(SemanticEntity(
                text=match.group(1),
                entity_type='quoted_term',
                confidence=0.7,
                start_pos=match.start(1),
                end_pos=match.end(1)
            ))
        
        # Extract capitalized words (likely proper nouns)
        proper_noun_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(proper_noun_pattern, query_text):
            # Skip common words
            if match.group().lower() not in ['The', 'This', 'That', 'When', 'Where', 'What', 'How']:
                entities.append(SemanticEntity(
                    text=match.group(),
                    entity_type='proper_noun',
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end()
                ))
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        for match in re.finditer(number_pattern, query_text):
            entities.append(SemanticEntity(
                text=match.group(),
                entity_type='number',
                confidence=0.5,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return entities
    
    def _extract_relations(self, entities: List[SemanticEntity], 
                         query_text: str) -> List[SemanticRelation]:
        """Extract semantic relations between entities"""
        relations = []
        
        # Simple relation extraction based on proximity and connecting words
        relation_patterns = {
            'temporal': ['before', 'after', 'during', 'when', 'since', 'until'],
            'causal': ['because', 'caused', 'due to', 'resulted in', 'led to'],
            'spatial': ['in', 'at', 'near', 'above', 'below', 'beside'],
            'association': ['with', 'and', 'related to', 'associated with'],
            'comparison': ['like', 'similar to', 'different from', 'compared to']
        }
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Find text between entities
                start_pos = min(entity1.end_pos, entity2.end_pos)
                end_pos = max(entity1.start_pos, entity2.start_pos)
                
                if start_pos < end_pos:
                    between_text = query_text[start_pos:end_pos].lower()
                    
                    # Check for relation patterns
                    for relation_type, patterns in relation_patterns.items():
                        for pattern in patterns:
                            if pattern in between_text:
                                relations.append(SemanticRelation(
                                    subject=entity1,
                                    predicate=relation_type,
                                    object=entity2,
                                    confidence=0.6,
                                    metadata={'pattern': pattern, 'between_text': between_text}
                                ))
                                break
        
        return relations
    
    def _analyze_temporal_aspects(self, query: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """Analyze temporal aspects of the query"""
        aspects = {}
        text_lower = query_text.lower()
        
        # Check for temporal keywords
        for aspect_type, keywords in self.vocabulary.TEMPORAL_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                aspects[aspect_type] = found_keywords
        
        # Check for explicit time ranges
        if any(field in query for field in ['start_time', 'end_time', 'time_range']):
            aspects['explicit_time_range'] = True
        
        # Check for relative time expressions
        relative_patterns = [
            r'\b\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)\s*ago\b',
            r'\blast\s+\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)\b',
            r'\bnext\s+\d+\s*(?:minutes?|hours?|days?|weeks?|months?|years?)\b'
        ]
        
        for pattern in relative_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                aspects['relative_expressions'] = matches
        
        return aspects
    
    def _analyze_spatial_aspects(self, query: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """Analyze spatial aspects of the query"""
        aspects = {}
        text_lower = query_text.lower()
        
        # Check for spatial keywords
        for aspect_type, keywords in self.vocabulary.SPATIAL_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                aspects[aspect_type] = found_keywords
        
        # Check for explicit location fields
        if any(field in query for field in ['location', 'place', 'coordinates']):
            aspects['explicit_location'] = True
        
        return aspects
    
    def _analyze_emotional_aspects(self, query: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """Analyze emotional aspects of the query"""
        aspects = {}
        text_lower = query_text.lower()
        
        # Check for emotional keywords
        for aspect_type, keywords in self.vocabulary.EMOTIONAL_KEYWORDS.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                aspects[aspect_type] = found_keywords
        
        # Simple sentiment analysis (positive/negative/neutral)
        positive_count = sum(1 for word in self.vocabulary.EMOTIONAL_KEYWORDS['positive'] 
                           if word in text_lower)
        negative_count = sum(1 for word in self.vocabulary.EMOTIONAL_KEYWORDS['negative'] 
                           if word in text_lower)
        
        if positive_count > negative_count:
            aspects['sentiment'] = 'positive'
        elif negative_count > positive_count:
            aspects['sentiment'] = 'negative'
        else:
            aspects['sentiment'] = 'neutral'
        
        aspects['emotional_intensity'] = positive_count + negative_count
        
        return aspects
    
    def _calculate_confidence_score(self, intent: SemanticIntent, complexity: QueryComplexity,
                                  domains: List[MemoryDomain], entities: List[SemanticEntity],
                                  relations: List[SemanticRelation]) -> float:
        """Calculate overall confidence score for the semantic analysis"""
        score = 0.0
        
        # Intent confidence (base score)
        score += 0.7  # Assume reasonable intent classification
        
        # Entity confidence
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            score += 0.2 * avg_entity_confidence
        else:
            score += 0.1  # Some penalty for no entities
        
        # Relation confidence
        if relations:
            avg_relation_confidence = sum(r.confidence for r in relations) / len(relations)
            score += 0.1 * avg_relation_confidence
        
        # Domain confidence (based on number of identified domains)
        if len(domains) > 0:
            domain_confidence = min(len(domains) / 3, 1.0)  # Max confidence at 3 domains
            score *= (0.8 + 0.2 * domain_confidence)
        
        return min(score, 1.0)
    
    def _generate_optimization_hints(self, intent: SemanticIntent, complexity: QueryComplexity,
                                   domains: List[MemoryDomain], entities: List[SemanticEntity],
                                   temporal_aspects: Dict[str, Any]) -> List[str]:
        """Generate optimization hints based on semantic analysis"""
        hints = []
        
        # Intent-based hints
        if intent == SemanticIntent.SEARCH_SIMILARITY:
            hints.append("Consider using vector similarity search for semantic matching")
        elif intent == SemanticIntent.TEMPORAL_QUERY:
            hints.append("Use temporal indexes for time-based queries")
        elif intent == SemanticIntent.PATTERN_QUERY:
            hints.append("Consider pattern matching optimizations and result caching")
        
        # Complexity-based hints
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            hints.append("Break complex query into smaller, parallelizable sub-queries")
            hints.append("Consider intermediate result caching for complex operations")
        
        # Domain-based hints
        if MemoryDomain.EPISODIC in domains:
            hints.append("Use temporal partitioning for episodic memory queries")
        if MemoryDomain.SEMANTIC in domains:
            hints.append("Leverage semantic indexes for concept-based queries")
        
        # Entity-based hints
        if len(entities) > 5:
            hints.append("Pre-process entities to reduce resolution overhead")
        
        # Temporal hints
        if temporal_aspects:
            if 'relative_time' in temporal_aspects:
                hints.append("Convert relative time expressions to absolute ranges")
            if 'frequency' in temporal_aspects:
                hints.append("Use frequency-aware caching strategies")
        
        return hints
    
    async def _decompose_complex_query(self, semantics: QuerySemantics) -> Optional[List[Dict[str, Any]]]:
        """Decompose complex query into simpler sub-queries"""
        if semantics.complexity not in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            return None
        
        sub_queries = []
        original = semantics.original_query
        
        # Separate by domains
        if len(semantics.domains) > 1:
            for domain in semantics.domains:
                sub_query = original.copy()
                sub_query['memory_types'] = [domain.value]
                sub_query['_sub_query_for'] = domain.value
                sub_queries.append(sub_query)
        
        # Separate temporal ranges
        if semantics.temporal_aspects and 'explicit_time_range' in semantics.temporal_aspects:
            # Would implement time range splitting
            pass
        
        return sub_queries if sub_queries else None
    
    async def _rewrite_for_vector_search(self, semantics: QuerySemantics) -> Optional[Dict[str, Any]]:
        """Rewrite query to use vector similarity search"""
        if semantics.intent != SemanticIntent.SEARCH_SIMILARITY:
            return None
        
        original = semantics.original_query
        vector_query = original.copy()
        
        # Add vector search parameters
        vector_query['search_type'] = 'vector_similarity'
        vector_query['use_embeddings'] = True
        
        # Extract text for embedding
        query_text = self._extract_query_text(original)
        if query_text:
            vector_query['embedding_text'] = query_text
        
        return vector_query
    
    async def _rewrite_for_temporal_optimization(self, semantics: QuerySemantics) -> Optional[Dict[str, Any]]:
        """Rewrite query for temporal optimization"""
        if not semantics.temporal_aspects:
            return None
        
        original = semantics.original_query
        temporal_query = original.copy()
        
        # Add temporal optimization hints
        temporal_query['use_temporal_index'] = True
        temporal_query['temporal_optimization'] = True
        
        # Convert relative times to absolute
        if 'relative_expressions' in semantics.temporal_aspects:
            temporal_query['_relative_converted'] = True
        
        return temporal_query
    
    async def _rewrite_for_filter_pushdown(self, semantics: QuerySemantics) -> Optional[Dict[str, Any]]:
        """Rewrite query to push filters closer to data sources"""
        if not semantics.entities:
            return None
        
        original = semantics.original_query
        filter_query = original.copy()
        
        # Add filter pushdown hints
        filter_query['push_down_filters'] = True
        filter_query['early_filtering'] = True
        
        # Extract filterable entities
        filterable_entities = [
            e for e in semantics.entities 
            if e.entity_type in ['date', 'time', 'number', 'quoted_term']
        ]
        
        if filterable_entities:
            filter_query['_filterable_entities'] = [e.text for e in filterable_entities]
        
        return filter_query
    
    def _calculate_pattern_benefit(self, intent: SemanticIntent, frequency: int) -> float:
        """Calculate optimization benefit for a semantic pattern"""
        base_benefit = frequency * 0.1  # Base benefit from frequency
        
        # Intent-specific multipliers
        intent_multipliers = {
            SemanticIntent.SEARCH_SIMILARITY: 1.5,  # High benefit for similarity
            SemanticIntent.TEMPORAL_QUERY: 1.3,     # Good benefit for temporal
            SemanticIntent.RETRIEVE_MEMORY: 1.2,    # Standard retrieval
            SemanticIntent.ANALYZE_MEMORY: 1.4,     # Analysis benefits from caching
        }
        
        multiplier = intent_multipliers.get(intent, 1.0)
        return base_benefit * multiplier
    
    async def _update_semantic_patterns(self, semantics: QuerySemantics):
        """Update semantic patterns based on new analysis"""
        # This would update the pattern cache with new observations
        pattern_key = f"{semantics.intent.value}_{len(semantics.domains)}"
        
        if pattern_key not in self.pattern_cache:
            self.pattern_cache[pattern_key] = {
                'count': 0,
                'examples': [],
                'last_seen': None
            }
        
        self.pattern_cache[pattern_key]['count'] += 1
        self.pattern_cache[pattern_key]['last_seen'] = datetime.utcnow()
        
        # Add example (limit to 5)
        if len(self.pattern_cache[pattern_key]['examples']) < 5:
            self.pattern_cache[pattern_key]['examples'].append(
                str(semantics.original_query)[:100]
            )
    
    async def clear_cache(self, max_age_hours: int = 24):
        """Clear old cache entries"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Clear analysis cache (simple approach - clear all)
        # In production, would check timestamps
        if len(self.analysis_cache) > 1000:
            self.analysis_cache.clear()
        
        # Clear old patterns
        self.semantic_patterns = [
            p for p in self.semantic_patterns 
            if p.last_seen > cutoff_time
        ]