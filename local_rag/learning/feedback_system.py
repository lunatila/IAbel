"""
Feedback Learning System - Sistema de aprendizado com feedback do usuário
Aprende com feedback positivo/negativo para melhorar respostas futuras
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import logging

@dataclass
class UserFeedback:
    """
    Feedback do usuário sobre uma resposta
    """
    feedback_id: str
    response_id: str
    user_id: str
    question: str
    response_text: str
    rating: int  # 1-5 scale
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'rating', 'detailed'
    feedback_text: Optional[str]
    aspects: Dict[str, int]  # accuracy, relevance, completeness, clarity
    timestamp: datetime
    source_quality: Optional[int]  # Rating of source relevance
    citation_quality: Optional[int]  # Rating of citation accuracy
    metadata: Dict[str, Any]

@dataclass
class LearningPattern:
    """
    Padrão aprendido do feedback
    """
    pattern_id: str
    pattern_type: str  # 'query_preference', 'source_preference', 'style_preference'
    conditions: Dict[str, Any]
    adjustments: Dict[str, Any]
    confidence: float
    support_count: int  # Number of feedback instances supporting this pattern
    created_at: datetime
    last_updated: datetime

@dataclass
class QueryPerformance:
    """
    Performance histórico de tipos de query
    """
    query_pattern: str
    avg_rating: float
    total_feedback: int
    success_rate: float
    common_issues: List[str]
    preferred_sources: List[str]
    optimal_length: int

class FeedbackLearningSystem:
    """
    Sistema avançado de aprendizado com feedback para RAG
    """
    
    def __init__(self, db_path: str = "data/feedback.db"):
        """
        Initialize feedback learning system
        
        Args:
            db_path: Path to SQLite database for storing feedback
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Learning parameters
        self.min_feedback_for_pattern = 5  # Minimum feedback to establish pattern
        self.confidence_threshold = 0.7    # Minimum confidence for applying adjustments
        self.learning_rate = 0.1           # How quickly to adapt to new feedback
        
        # Cache for learned patterns
        self._pattern_cache = {}
        self._last_cache_update = datetime.now()
        self._cache_ttl = timedelta(hours=1)
        
        print(f"✅ Feedback Learning System initialized (DB: {db_path})")
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    response_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    feedback_type TEXT NOT NULL,
                    feedback_text TEXT,
                    aspects TEXT,  -- JSON
                    timestamp TEXT NOT NULL,
                    source_quality INTEGER,
                    citation_quality INTEGER,
                    metadata TEXT  -- JSON
                )
            ''')
            
            # Learning patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,  -- JSON
                    adjustments TEXT NOT NULL,  -- JSON
                    confidence REAL NOT NULL,
                    support_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Query performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_performance (
                    query_pattern TEXT PRIMARY KEY,
                    avg_rating REAL NOT NULL,
                    total_feedback INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    common_issues TEXT,  -- JSON
                    preferred_sources TEXT,  -- JSON
                    optimal_length INTEGER
                )
            ''')
            
            conn.commit()
    
    def record_feedback(self, 
                       response_id: str,
                       user_id: str,
                       question: str,
                       response_text: str,
                       rating: int,
                       feedback_type: str = "rating",
                       feedback_text: Optional[str] = None,
                       aspects: Optional[Dict[str, int]] = None,
                       source_quality: Optional[int] = None,
                       citation_quality: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record user feedback and trigger learning
        
        Args:
            response_id: ID of the response being rated
            user_id: ID of the user providing feedback
            question: Original question
            response_text: Generated response
            rating: 1-5 rating
            feedback_type: Type of feedback
            feedback_text: Optional text feedback
            aspects: Ratings for different aspects
            source_quality: Rating of source quality
            citation_quality: Rating of citation quality
            metadata: Additional metadata
            
        Returns:
            feedback_id: Unique ID for this feedback
        """
        feedback_id = str(uuid.uuid4())
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            response_id=response_id,
            user_id=user_id,
            question=question,
            response_text=response_text,
            rating=rating,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            aspects=aspects or {},
            timestamp=datetime.now(),
            source_quality=source_quality,
            citation_quality=citation_quality,
            metadata=metadata or {}
        )
        
        # Store in database
        self._store_feedback(feedback)
        
        # Trigger learning from new feedback
        self._learn_from_feedback(feedback)
        
        # Update query performance
        self._update_query_performance(question, rating, feedback_text)
        
        return feedback_id
    
    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (
                    feedback_id, response_id, user_id, question, response_text,
                    rating, feedback_type, feedback_text, aspects, timestamp,
                    source_quality, citation_quality, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.response_id,
                feedback.user_id,
                feedback.question,
                feedback.response_text,
                feedback.rating,
                feedback.feedback_type,
                feedback.feedback_text,
                json.dumps(feedback.aspects),
                feedback.timestamp.isoformat(),
                feedback.source_quality,
                feedback.citation_quality,
                json.dumps(feedback.metadata)
            ))
            
            conn.commit()
    
    def _learn_from_feedback(self, feedback: UserFeedback):
        """Learn patterns from new feedback"""
        # Extract features from the feedback
        features = self._extract_features(feedback)
        
        # Find or create relevant patterns
        patterns = self._identify_patterns(features, feedback)
        
        # Update pattern strengths
        for pattern in patterns:
            self._update_pattern(pattern, feedback)
    
    def _extract_features(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Extract learning features from feedback"""
        features = {
            'question_length': len(feedback.question.split()),
            'response_length': len(feedback.response_text.split()),
            'question_type': self._classify_question_type(feedback.question),
            'rating': feedback.rating,
            'has_technical_terms': self._has_technical_terms(feedback.question),
            'complexity_level': self._estimate_complexity(feedback.question),
            'user_satisfaction': feedback.rating >= 4,  # 4-5 considered satisfied
            'aspects': feedback.aspects,
            'source_quality': feedback.source_quality,
            'citation_quality': feedback.citation_quality
        }
        
        return features
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question into types"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['o que é', 'what is', 'define', 'definir']):
            return 'definition'
        elif any(word in question_lower for word in ['como', 'how', 'processo']):
            return 'process'
        elif any(word in question_lower for word in ['compare', 'diferença', 'difference']):
            return 'comparison'
        elif any(word in question_lower for word in ['vantagem', 'desvantagem', 'advantage']):
            return 'evaluation'
        elif any(word in question_lower for word in ['exemplo', 'example', 'case']):
            return 'example'
        else:
            return 'general'
    
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical terms"""
        technical_terms = {
            'insim', 'permeabilidade', 'permeability', 'simulação', 'simulation',
            'reservatório', 'reservoir', 'waterflooding', 'eclipse', 'cmg'
        }
        text_lower = text.lower()
        return any(term in text_lower for term in technical_terms)
    
    def _estimate_complexity(self, question: str) -> str:
        """Estimate question complexity"""
        word_count = len(question.split())
        technical_count = sum(1 for word in question.lower().split() 
                             if word in ['insim', 'simulação', 'modelo', 'permeabilidade'])
        
        if word_count < 5:
            return 'simple'
        elif word_count < 15 and technical_count <= 1:
            return 'moderate'
        else:
            return 'complex'
    
    def _identify_patterns(self, features: Dict[str, Any], feedback: UserFeedback) -> List[LearningPattern]:
        """Identify relevant learning patterns"""
        patterns = []
        
        # Query type preference pattern
        if features['rating'] >= 4:  # Good feedback
            pattern = LearningPattern(
                pattern_id=f"query_pref_{features['question_type']}",
                pattern_type='query_preference',
                conditions={'question_type': features['question_type']},
                adjustments={
                    'preferred_response_length': features['response_length'],
                    'include_technical_detail': features['has_technical_terms'],
                    'complexity_level': features['complexity_level']
                },
                confidence=0.5,  # Initial confidence
                support_count=1,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            patterns.append(pattern)
        
        # Source quality pattern
        if feedback.source_quality is not None and feedback.source_quality >= 4:
            pattern = LearningPattern(
                pattern_id=f"source_pref_{feedback.response_id}",
                pattern_type='source_preference',
                conditions={'question_type': features['question_type']},
                adjustments={
                    'boost_similar_sources': True,
                    'source_confidence_threshold': 0.8
                },
                confidence=0.6,
                support_count=1,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            patterns.append(pattern)
        
        return patterns
    
    def _update_pattern(self, pattern: LearningPattern, feedback: UserFeedback):
        """Update pattern based on new feedback"""
        # Check if pattern already exists in database
        existing_pattern = self._get_pattern(pattern.pattern_id)
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.support_count += 1
            existing_pattern.confidence = min(1.0, existing_pattern.confidence + self.learning_rate)
            existing_pattern.last_updated = datetime.now()
            
            # Update adjustments based on new feedback
            if feedback.rating >= 4:  # Positive feedback
                self._reinforce_pattern(existing_pattern, pattern.adjustments)
            else:  # Negative feedback
                self._weaken_pattern(existing_pattern)
            
            self._store_pattern(existing_pattern)
        else:
            # Create new pattern
            self._store_pattern(pattern)
    
    def _get_pattern(self, pattern_id: str) -> Optional[LearningPattern]:
        """Get pattern from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM learning_patterns WHERE pattern_id = ?
            ''', (pattern_id,))
            
            row = cursor.fetchone()
            if row:
                return LearningPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    conditions=json.loads(row[2]),
                    adjustments=json.loads(row[3]),
                    confidence=row[4],
                    support_count=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    last_updated=datetime.fromisoformat(row[7])
                )
        return None
    
    def _store_pattern(self, pattern: LearningPattern):
        """Store pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_patterns (
                    pattern_id, pattern_type, conditions, adjustments,
                    confidence, support_count, created_at, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                pattern.pattern_type,
                json.dumps(pattern.conditions),
                json.dumps(pattern.adjustments),
                pattern.confidence,
                pattern.support_count,
                pattern.created_at.isoformat(),
                pattern.last_updated.isoformat()
            ))
            
            conn.commit()
    
    def _reinforce_pattern(self, pattern: LearningPattern, new_adjustments: Dict[str, Any]):
        """Reinforce pattern with positive feedback"""
        # Merge adjustments with weight towards new successful patterns
        for key, value in new_adjustments.items():
            if key in pattern.adjustments:
                if isinstance(value, (int, float)):
                    pattern.adjustments[key] = (pattern.adjustments[key] * 0.8 + value * 0.2)
                else:
                    pattern.adjustments[key] = value  # Replace non-numeric values
            else:
                pattern.adjustments[key] = value
    
    def _weaken_pattern(self, pattern: LearningPattern):
        """Weaken pattern with negative feedback"""
        pattern.confidence = max(0.1, pattern.confidence - self.learning_rate * 0.5)
    
    def _update_query_performance(self, question: str, rating: int, feedback_text: Optional[str]):
        """Update query performance statistics"""
        query_pattern = self._classify_question_type(question)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get existing performance
            cursor.execute('''
                SELECT * FROM query_performance WHERE query_pattern = ?
            ''', (query_pattern,))
            
            row = cursor.fetchone()
            
            if row:
                # Update existing
                avg_rating = (row[1] * row[2] + rating) / (row[2] + 1)
                total_feedback = row[2] + 1
                success_rate = (row[3] * (row[2] - 1) + (1 if rating >= 4 else 0)) / row[2]
                
                cursor.execute('''
                    UPDATE query_performance SET
                    avg_rating = ?, total_feedback = ?, success_rate = ?
                    WHERE query_pattern = ?
                ''', (avg_rating, total_feedback, success_rate, query_pattern))
            else:
                # Create new
                cursor.execute('''
                    INSERT INTO query_performance (
                        query_pattern, avg_rating, total_feedback, success_rate,
                        common_issues, preferred_sources, optimal_length
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    query_pattern, float(rating), 1, 1.0 if rating >= 4 else 0.0,
                    json.dumps([]), json.dumps([]), len(question.split())
                ))
            
            conn.commit()
    
    def get_adjustment_recommendations(self, question: str, context_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get recommendations for adjusting RAG system based on learned patterns"""
        question_type = self._classify_question_type(question)
        features = {
            'question_type': question_type,
            'has_technical_terms': self._has_technical_terms(question),
            'complexity_level': self._estimate_complexity(question)
        }
        
        recommendations = {
            'response_adjustments': {},
            'source_adjustments': {},
            'citation_adjustments': {},
            'confidence': 0.0
        }
        
        # Get relevant patterns
        patterns = self._get_applicable_patterns(features)
        
        if patterns:
            total_confidence = 0
            
            for pattern in patterns:
                if pattern.confidence >= self.confidence_threshold:
                    # Apply pattern adjustments
                    for key, value in pattern.adjustments.items():
                        if 'length' in key:
                            recommendations['response_adjustments'][key] = value
                        elif 'source' in key:
                            recommendations['source_adjustments'][key] = value
                        elif 'citation' in key:
                            recommendations['citation_adjustments'][key] = value
                        else:
                            recommendations['response_adjustments'][key] = value
                    
                    total_confidence += pattern.confidence
            
            recommendations['confidence'] = total_confidence / len(patterns) if patterns else 0
        
        return recommendations
    
    def _get_applicable_patterns(self, features: Dict[str, Any]) -> List[LearningPattern]:
        """Get patterns applicable to current features"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM learning_patterns 
                WHERE confidence >= ? AND support_count >= ?
            ''', (self.confidence_threshold, self.min_feedback_for_pattern))
            
            for row in cursor.fetchall():
                pattern = LearningPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    conditions=json.loads(row[2]),
                    adjustments=json.loads(row[3]),
                    confidence=row[4],
                    support_count=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    last_updated=datetime.fromisoformat(row[7])
                )
                
                # Check if pattern conditions match current features
                if self._pattern_matches(pattern.conditions, features):
                    patterns.append(pattern)
        
        return patterns
    
    def _pattern_matches(self, conditions: Dict[str, Any], features: Dict[str, Any]) -> bool:
        """Check if pattern conditions match current features"""
        for key, value in conditions.items():
            if key in features and features[key] == value:
                return True
        return len(conditions) == 0  # Empty conditions match everything
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute('SELECT COUNT(*), AVG(rating) FROM feedback')
            total_feedback, avg_rating = cursor.fetchone()
            
            # Rating distribution
            cursor.execute('SELECT rating, COUNT(*) FROM feedback GROUP BY rating')
            rating_dist = dict(cursor.fetchall())
            
            # Question type performance
            cursor.execute('''
                SELECT question, rating, feedback_type 
                FROM feedback 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            recent_feedback = cursor.fetchall()
            
            # Pattern stats
            cursor.execute('SELECT COUNT(*), AVG(confidence) FROM learning_patterns')
            pattern_count, avg_confidence = cursor.fetchone()
            
        return {
            'total_feedback': total_feedback or 0,
            'average_rating': round(avg_rating, 2) if avg_rating else 0,
            'rating_distribution': rating_dist,
            'learned_patterns': pattern_count or 0,
            'average_pattern_confidence': round(avg_confidence, 3) if avg_confidence else 0,
            'recent_feedback_count': len(recent_feedback),
            'success_rate': len([f for f in recent_feedback if f[1] >= 4]) / len(recent_feedback) if recent_feedback else 0
        }


# Global feedback learning system instance
_global_feedback_system = None

def get_feedback_system(db_path: str = "data/feedback.db") -> FeedbackLearningSystem:
    """
    Get or create global feedback learning system instance
    """
    global _global_feedback_system
    if _global_feedback_system is None:
        _global_feedback_system = FeedbackLearningSystem(db_path)
    return _global_feedback_system