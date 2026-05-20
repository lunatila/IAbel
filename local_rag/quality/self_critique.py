"""
Self-Critique System - Automated quality assessment and validation for RAG responses
Evaluates answer quality, relevance, factual consistency, and completeness
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CritiqueResult:
    """
    Result of self-critique analysis
    """
    overall_score: float
    relevance_score: float
    factual_score: float
    completeness_score: float
    clarity_score: float
    technical_accuracy_score: float
    confidence_score: float
    issues_found: List[str]
    suggestions: List[str]
    validated: bool
    critique_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'relevance_score': self.relevance_score,
            'factual_score': self.factual_score,
            'completeness_score': self.completeness_score,
            'clarity_score': self.clarity_score,
            'technical_accuracy_score': self.technical_accuracy_score,
            'confidence_score': self.confidence_score,
            'issues_found': self.issues_found,
            'suggestions': self.suggestions,
            'validated': self.validated,
            'critique_details': self.critique_details
        }


class AnswerRelevanceAnalyzer:
    """
    Analyzes how well the answer addresses the question
    """
    
    def __init__(self, embedder_model: Optional[SentenceTransformer] = None):
        if embedder_model is None:
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedder = embedder_model
    
    def analyze_relevance(self, question: str, answer: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze how relevant the answer is to the question
        
        Args:
            question: User question
            answer: Generated answer
            
        Returns:
            Relevance score (0-1) and analysis details
        """
        details = {}
        
        # Semantic similarity between question and answer
        q_embedding = self.embedder.encode([question])
        a_embedding = self.embedder.encode([answer])
        semantic_similarity = cosine_similarity(q_embedding, a_embedding)[0][0]
        
        # Keyword overlap analysis
        q_keywords = self._extract_keywords(question)
        a_keywords = self._extract_keywords(answer)
        
        if q_keywords:
            keyword_overlap = len(set(q_keywords) & set(a_keywords)) / len(set(q_keywords))
        else:
            keyword_overlap = 0.0
        
        # Question type analysis
        question_type = self._classify_question_type(question)
        answer_structure = self._analyze_answer_structure(answer, question_type)
        
        # Calculate relevance score
        relevance_score = (
            semantic_similarity * 0.4 +
            keyword_overlap * 0.3 +
            answer_structure * 0.3
        )
        
        details = {
            'semantic_similarity': semantic_similarity,
            'keyword_overlap': keyword_overlap,
            'question_keywords': q_keywords,
            'answer_keywords': a_keywords,
            'question_type': question_type,
            'answer_structure_score': answer_structure
        }
        
        return min(1.0, relevance_score), details
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        stop_words = {
            'que', 'como', 'qual', 'onde', 'quando', 'por', 'para', 'com', 'sem',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize technical terms
        technical_terms = [
            'insim', 'permeabilidade', 'permeability', 'porosidade', 'porosity',
            'saturação', 'saturation', 'viscosidade', 'viscosity', 'densidade',
            'density', 'simulação', 'simulation', 'reservatório', 'reservoir'
        ]
        
        prioritized = []
        for term in technical_terms:
            if term in keywords:
                prioritized.append(term)
        
        # Add non-technical keywords
        for word in keywords:
            if word not in prioritized:
                prioritized.append(word)
        
        return prioritized[:10]  # Top 10 keywords
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['o que é', 'what is', 'define', 'definição']):
            return 'definition'
        elif any(word in question_lower for word in ['como', 'how', 'method', 'processo']):
            return 'process'
        elif any(word in question_lower for word in ['por que', 'why', 'porque', 'razão']):
            return 'explanation'
        elif any(word in question_lower for word in ['qual', 'which', 'que tipo']):
            return 'selection'
        elif any(word in question_lower for word in ['onde', 'where', 'quando', 'when']):
            return 'factual'
        else:
            return 'general'
    
    def _analyze_answer_structure(self, answer: str, question_type: str) -> float:
        """Analyze if answer structure matches question type"""
        answer_lower = answer.lower()
        
        structure_indicators = {
            'definition': ['é', 'significa', 'refere-se', 'definido como', 'is defined'],
            'process': ['primeiro', 'segundo', 'etapa', 'passo', 'step', 'process'],
            'explanation': ['porque', 'devido', 'razão', 'causa', 'because', 'due to'],
            'selection': ['melhor', 'recomendado', 'ideal', 'best', 'recommended'],
            'factual': ['localizado', 'ocorre', 'happens', 'located'],
            'general': []
        }
        
        indicators = structure_indicators.get(question_type, [])
        if not indicators:
            return 0.7  # Neutral score for general questions
        
        matches = sum(1 for indicator in indicators if indicator in answer_lower)
        return min(1.0, matches / len(indicators) + 0.3)  # Base score + matches


class FactualConsistencyChecker:
    """
    Checks factual consistency between sources and answer
    """
    
    def check_consistency(self, answer: str, sources: List[str]) -> Tuple[float, Dict[str, Any]]:
        """
        Check factual consistency between answer and source documents
        
        Args:
            answer: Generated answer
            sources: Source documents used for generation
            
        Returns:
            Consistency score (0-1) and analysis details
        """
        if not sources:
            return 0.5, {'reason': 'no_sources', 'details': 'No sources provided for verification'}
        
        details = {}
        
        # Extract claims from answer
        answer_claims = self._extract_claims(answer)
        
        # Check each claim against sources
        verified_claims = 0
        claim_details = []
        
        for claim in answer_claims:
            is_supported = self._verify_claim_in_sources(claim, sources)
            if is_supported:
                verified_claims += 1
            
            claim_details.append({
                'claim': claim,
                'supported': is_supported
            })
        
        # Calculate consistency score
        if answer_claims:
            consistency_score = verified_claims / len(answer_claims)
        else:
            consistency_score = 0.8  # Neutral if no specific claims
        
        # Check for contradictions
        contradictions = self._detect_contradictions(answer, sources)
        
        # Penalize for contradictions
        if contradictions:
            consistency_score *= (1 - len(contradictions) * 0.2)
        
        details = {
            'total_claims': len(answer_claims),
            'verified_claims': verified_claims,
            'claim_details': claim_details,
            'contradictions_found': contradictions,
            'consistency_ratio': verified_claims / len(answer_claims) if answer_claims else 0
        }
        
        return max(0.0, min(1.0, consistency_score)), details
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split by sentences and filter for factual statements
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Too short
                continue
            
            # Look for factual indicators
            factual_indicators = [
                'é', 'são', 'tem', 'possui', 'caracteriza-se', 'define-se',
                'is', 'are', 'has', 'have', 'characterized by', 'defined as'
            ]
            
            if any(indicator in sentence.lower() for indicator in factual_indicators):
                claims.append(sentence)
        
        return claims[:5]  # Limit to 5 main claims
    
    def _verify_claim_in_sources(self, claim: str, sources: List[str]) -> bool:
        """Verify if a claim is supported by sources"""
        claim_keywords = self._extract_keywords_simple(claim)
        
        for source in sources:
            source_keywords = self._extract_keywords_simple(source)
            
            # Check keyword overlap
            overlap = len(set(claim_keywords) & set(source_keywords))
            overlap_ratio = overlap / len(claim_keywords) if claim_keywords else 0
            
            if overlap_ratio > 0.5:  # 50% keyword overlap
                return True
        
        return False
    
    def _extract_keywords_simple(self, text: str) -> List[str]:
        """Simple keyword extraction"""
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop_words = {'que', 'para', 'com', 'uma', 'dos', 'das', 'the', 'and', 'for', 'with'}
        return [word for word in words if word not in stop_words]
    
    def _detect_contradictions(self, answer: str, sources: List[str]) -> List[str]:
        """Detect potential contradictions between answer and sources"""
        contradictions = []
        
        # Simple contradiction detection patterns
        contradiction_patterns = [
            (r'não é', r'é'),  # "não é" vs "é"
            (r'não tem', r'tem'),  # "não tem" vs "tem"
            (r'impossível', r'possível'),  # "impossível" vs "possível"
            (r'never', r'always'),
            (r'cannot', r'can')
        ]
        
        answer_lower = answer.lower()
        sources_text = ' '.join(sources).lower()
        
        for neg_pattern, pos_pattern in contradiction_patterns:
            if re.search(neg_pattern, answer_lower) and re.search(pos_pattern, sources_text):
                contradictions.append(f"Answer contains '{neg_pattern}' but sources suggest '{pos_pattern}'")
            elif re.search(pos_pattern, answer_lower) and re.search(neg_pattern, sources_text):
                contradictions.append(f"Answer contains '{pos_pattern}' but sources suggest '{neg_pattern}'")
        
        return contradictions


class TechnicalAccuracyValidator:
    """
    Validates technical accuracy for engineering domain
    """
    
    def __init__(self):
        # Common technical validation rules for reservoir engineering
        self.technical_rules = {
            'units': {
                'permeability': ['md', 'millidarcy', 'darcy', 'm²', 'm2'],
                'pressure': ['psi', 'bar', 'pa', 'pascal', 'atm'],
                'viscosity': ['cp', 'centipoise', 'pa.s', 'pas'],
                'density': ['kg/m³', 'g/cm³', 'lb/ft³'],
                'rate': ['bbl/d', 'bpd', 'm³/d', 'scf/d']
            },
            'typical_ranges': {
                'permeability': (0.1, 10000),  # mD
                'porosity': (0.05, 0.35),  # fraction
                'oil_viscosity': (0.1, 1000),  # cp
                'water_saturation': (0.0, 1.0)  # fraction
            },
            'technical_terms': [
                'INSIM', 'INSIM-FT', 'Eclipse', 'CMG', 'simulação', 'simulation',
                'permeabilidade', 'permeability', 'porosidade', 'porosity',
                'saturação', 'saturation', 'viscosidade', 'viscosity'
            ]
        }
    
    def validate_technical_accuracy(self, answer: str, context: str = "") -> Tuple[float, Dict[str, Any]]:
        """
        Validate technical accuracy of the answer
        
        Args:
            answer: Generated answer
            context: Context from source documents
            
        Returns:
            Technical accuracy score (0-1) and details
        """
        details = {}
        score_components = []
        
        # Check unit consistency
        unit_score, unit_details = self._check_unit_consistency(answer)
        score_components.append(unit_score)
        details['units'] = unit_details
        
        # Check numerical ranges
        range_score, range_details = self._check_numerical_ranges(answer)
        score_components.append(range_score)
        details['ranges'] = range_details
        
        # Check technical terminology usage
        term_score, term_details = self._check_technical_terms(answer)
        score_components.append(term_score)
        details['terminology'] = term_details
        
        # Check for common technical errors
        error_score, error_details = self._check_common_errors(answer)
        score_components.append(error_score)
        details['errors'] = error_details
        
        # Calculate overall technical accuracy
        overall_score = np.mean(score_components)
        
        return overall_score, details
    
    def _check_unit_consistency(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check if units are used correctly"""
        issues = []
        correct_usage = 0
        total_usage = 0
        
        for property_name, valid_units in self.technical_rules['units'].items():
            # Find mentions of the property
            property_pattern = r'\b' + property_name.replace('_', r'[\s_]') + r'\b'
            property_matches = re.findall(property_pattern, text, re.IGNORECASE)
            
            if property_matches:
                total_usage += len(property_matches)
                
                # Check if valid units are mentioned nearby
                for unit in valid_units:
                    unit_pattern = r'\b' + re.escape(unit) + r'\b'
                    if re.search(unit_pattern, text, re.IGNORECASE):
                        correct_usage += 1
                        break
                else:
                    issues.append(f"Property '{property_name}' mentioned without appropriate units")
        
        score = correct_usage / total_usage if total_usage > 0 else 1.0
        
        return score, {
            'correct_usage': correct_usage,
            'total_usage': total_usage,
            'issues': issues
        }
    
    def _check_numerical_ranges(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check if numerical values are within typical ranges"""
        issues = []
        values_checked = 0
        values_in_range = 0
        
        # Extract numbers with context
        number_pattern = r'(\w+)\s*[=:]\s*([\d.,]+)\s*(\w+)?'
        matches = re.findall(number_pattern, text, re.IGNORECASE)
        
        for property_name, value_str, unit in matches:
            try:
                value = float(value_str.replace(',', '.'))
                values_checked += 1
                
                # Check against typical ranges
                property_lower = property_name.lower()
                for range_property, (min_val, max_val) in self.technical_rules['typical_ranges'].items():
                    if range_property in property_lower:
                        if min_val <= value <= max_val:
                            values_in_range += 1
                        else:
                            issues.append(f"Value {value} for {property_name} outside typical range [{min_val}, {max_val}]")
                        break
                else:
                    values_in_range += 1  # Unknown property, assume correct
                    
            except ValueError:
                issues.append(f"Invalid numerical format: {value_str}")
        
        score = values_in_range / values_checked if values_checked > 0 else 1.0
        
        return score, {
            'values_checked': values_checked,
            'values_in_range': values_in_range,
            'issues': issues
        }
    
    def _check_technical_terms(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check correct usage of technical terms"""
        text_lower = text.lower()
        terms_found = []
        
        for term in self.technical_rules['technical_terms']:
            if term.lower() in text_lower:
                terms_found.append(term)
        
        # Score based on technical term density and correctness
        term_density = len(terms_found) / len(text.split()) if text.split() else 0
        
        # Reasonable density is between 0.05 and 0.2 (5% to 20% technical terms)
        if 0.05 <= term_density <= 0.2:
            score = 1.0
        elif term_density < 0.05:
            score = 0.7  # Too few technical terms
        else:
            score = 0.8  # Too many technical terms
        
        return score, {
            'terms_found': terms_found,
            'term_density': term_density,
            'density_assessment': 'optimal' if 0.05 <= term_density <= 0.2 else 
                                 'low' if term_density < 0.05 else 'high'
        }
    
    def _check_common_errors(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check for common technical errors"""
        errors = []
        
        # Common misconceptions and errors
        error_patterns = [
            (r'permeabilidade.*alta.*porosidade', 'High permeability does not always mean high porosity'),
            (r'viscosidade.*temperatura', 'Check viscosity-temperature relationship context'),
            (r'pressão.*profundidade.*linear', 'Pressure-depth relationship may not always be linear'),
        ]
        
        text_lower = text.lower()
        
        for pattern, error_msg in error_patterns:
            if re.search(pattern, text_lower):
                errors.append(error_msg)
        
        # Score inversely related to number of errors
        score = max(0.0, 1.0 - len(errors) * 0.3)
        
        return score, {
            'errors_found': errors,
            'error_count': len(errors)
        }


class SelfCritiqueSystem:
    """
    Complete self-critique system for RAG answer validation
    """
    
    def __init__(self, embedder_model: Optional[SentenceTransformer] = None):
        """
        Initialize self-critique system
        
        Args:
            embedder_model: SentenceTransformer for semantic analysis
        """
        self.relevance_analyzer = AnswerRelevanceAnalyzer(embedder_model)
        self.consistency_checker = FactualConsistencyChecker()
        self.technical_validator = TechnicalAccuracyValidator()
        
        # Quality thresholds
        self.thresholds = {
            'overall_min': 0.6,
            'relevance_min': 0.7,
            'factual_min': 0.6,
            'technical_min': 0.7,
            'completeness_min': 0.6,
            'clarity_min': 0.6
        }
    
    def critique_answer(self, 
                       question: str,
                       answer: str,
                       sources: List[str],
                       context: str = "") -> CritiqueResult:
        """
        Perform comprehensive critique of generated answer
        
        Args:
            question: Original user question
            answer: Generated answer
            sources: Source documents used
            context: Additional context
            
        Returns:
            Comprehensive critique result
        """
        print(f"🔍 Self-critique: Analyzing answer quality...")
        
        # Analyze relevance
        relevance_score, relevance_details = self.relevance_analyzer.analyze_relevance(question, answer)
        
        # Check factual consistency
        factual_score, factual_details = self.consistency_checker.check_consistency(answer, sources)
        
        # Validate technical accuracy
        technical_score, technical_details = self.technical_validator.validate_technical_accuracy(answer, context)
        
        # Assess completeness
        completeness_score, completeness_details = self._assess_completeness(question, answer)
        
        # Assess clarity
        clarity_score, clarity_details = self._assess_clarity(answer)
        
        # Calculate confidence based on source quality
        confidence_score = self._calculate_confidence(sources, relevance_score, factual_score)
        
        # Calculate overall score
        overall_score = (
            relevance_score * 0.25 +
            factual_score * 0.25 +
            technical_score * 0.20 +
            completeness_score * 0.15 +
            clarity_score * 0.15
        )
        
        # Identify issues and suggestions
        issues_found = []
        suggestions = []
        
        if relevance_score < self.thresholds['relevance_min']:
            issues_found.append("Answer relevance to question is low")
            suggestions.append("Improve focus on specific question aspects")
        
        if factual_score < self.thresholds['factual_min']:
            issues_found.append("Factual consistency with sources is questionable")
            suggestions.append("Verify claims against source documents")
        
        if technical_score < self.thresholds['technical_min']:
            issues_found.append("Technical accuracy needs improvement")
            suggestions.append("Check technical terms, units, and numerical values")
        
        if completeness_score < self.thresholds['completeness_min']:
            issues_found.append("Answer lacks completeness")
            suggestions.append("Address all aspects of the question")
        
        if clarity_score < self.thresholds['clarity_min']:
            issues_found.append("Answer clarity can be improved")
            suggestions.append("Use clearer language and better structure")
        
        # Determine if answer is validated
        validated = (
            overall_score >= self.thresholds['overall_min'] and
            relevance_score >= self.thresholds['relevance_min'] and
            factual_score >= self.thresholds['factual_min']
        )
        
        # Compile critique details
        critique_details = {
            'relevance_analysis': relevance_details,
            'factual_analysis': factual_details,
            'technical_analysis': technical_details,
            'completeness_analysis': completeness_details,
            'clarity_analysis': clarity_details,
            'thresholds_used': self.thresholds,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   Overall score: {overall_score:.2f}")
        print(f"   Validated: {'✅' if validated else '❌'}")
        print(f"   Issues found: {len(issues_found)}")
        
        return CritiqueResult(
            overall_score=round(overall_score, 3),
            relevance_score=round(relevance_score, 3),
            factual_score=round(factual_score, 3),
            completeness_score=round(completeness_score, 3),
            clarity_score=round(clarity_score, 3),
            technical_accuracy_score=round(technical_score, 3),
            confidence_score=round(confidence_score, 3),
            issues_found=issues_found,
            suggestions=suggestions,
            validated=validated,
            critique_details=critique_details
        )
    
    def _assess_completeness(self, question: str, answer: str) -> Tuple[float, Dict[str, Any]]:
        """Assess how completely the answer addresses the question"""
        # Extract question aspects
        question_aspects = self._extract_question_aspects(question)
        
        # Check which aspects are addressed in the answer
        addressed_aspects = 0
        answer_lower = answer.lower()
        
        for aspect in question_aspects:
            if any(keyword in answer_lower for keyword in aspect['keywords']):
                addressed_aspects += 1
        
        completeness_score = addressed_aspects / len(question_aspects) if question_aspects else 0.8
        
        # Length consideration (very short answers likely incomplete)
        if len(answer.split()) < 20:
            completeness_score *= 0.7
        
        details = {
            'total_aspects': len(question_aspects),
            'addressed_aspects': addressed_aspects,
            'question_aspects': question_aspects,
            'answer_length_words': len(answer.split())
        }
        
        return min(1.0, completeness_score), details
    
    def _extract_question_aspects(self, question: str) -> List[Dict[str, Any]]:
        """Extract different aspects of a complex question"""
        aspects = []
        
        # Look for multiple question indicators
        multi_question_patterns = [
            r'\be\s+(?:como|qual|onde|quando|por\s+que)',  # Portuguese
            r'\band\s+(?:how|what|where|when|why)',  # English
            r'[,;]\s*(?:como|qual|onde|quando|por\s+que)',  # Comma-separated
            r'[,;]\s*(?:how|what|where|when|why)'
        ]
        
        question_lower = question.lower()
        
        # If multiple aspects detected
        for pattern in multi_question_patterns:
            if re.search(pattern, question_lower):
                # Split question into parts
                parts = re.split(r'[,;]\s*(?:e\s+)?(?:and\s+)?', question)
                for part in parts:
                    keywords = [word for word in part.split() if len(word) > 3]
                    if keywords:
                        aspects.append({
                            'text': part.strip(),
                            'keywords': keywords[:5]
                        })
                return aspects
        
        # Single aspect question
        keywords = [word for word in question.split() if len(word) > 3]
        aspects.append({
            'text': question,
            'keywords': keywords[:5]
        })
        
        return aspects
    
    def _assess_clarity(self, answer: str) -> Tuple[float, Dict[str, Any]]:
        """Assess clarity and readability of the answer"""
        clarity_factors = []
        
        # Sentence length analysis
        sentences = re.split(r'[.!?]+', answer)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        if sentence_lengths:
            avg_sentence_length = np.mean(sentence_lengths)
            # Optimal sentence length is 15-25 words
            if 15 <= avg_sentence_length <= 25:
                clarity_factors.append(1.0)
            elif avg_sentence_length < 15:
                clarity_factors.append(0.8)  # Too short, might lack detail
            else:
                clarity_factors.append(max(0.5, 1.0 - (avg_sentence_length - 25) * 0.02))
        else:
            clarity_factors.append(0.5)
        
        # Structure indicators
        structure_indicators = ['primeiro', 'segundo', 'finalmente', 'portanto', 'além disso',
                               'first', 'second', 'finally', 'therefore', 'furthermore']
        
        structure_score = min(1.0, sum(1 for indicator in structure_indicators 
                                     if indicator in answer.lower()) * 0.2 + 0.6)
        clarity_factors.append(structure_score)
        
        # Technical jargon balance
        total_words = len(answer.split())
        technical_words = len([word for word in answer.split() 
                              if word.lower() in self.technical_validator.technical_rules['technical_terms']])
        
        jargon_ratio = technical_words / total_words if total_words > 0 else 0
        # Optimal technical ratio is 10-20%
        if 0.1 <= jargon_ratio <= 0.2:
            jargon_score = 1.0
        elif jargon_ratio < 0.1:
            jargon_score = 0.8
        else:
            jargon_score = max(0.6, 1.0 - (jargon_ratio - 0.2) * 2)
        
        clarity_factors.append(jargon_score)
        
        overall_clarity = np.mean(clarity_factors)
        
        details = {
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'structure_score': structure_score,
            'jargon_ratio': jargon_ratio,
            'clarity_factors': clarity_factors
        }
        
        return overall_clarity, details
    
    def _calculate_confidence(self, sources: List[str], relevance_score: float, factual_score: float) -> float:
        """Calculate confidence based on sources and other factors"""
        # Base confidence from source availability
        if not sources:
            base_confidence = 0.3
        elif len(sources) == 1:
            base_confidence = 0.6
        elif len(sources) <= 3:
            base_confidence = 0.8
        else:
            base_confidence = 0.9
        
        # Adjust by relevance and factual scores
        confidence = base_confidence * ((relevance_score + factual_score) / 2)
        
        return min(1.0, confidence)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the self-critique system"""
        return {
            'components': {
                'relevance_analyzer': True,
                'factual_consistency_checker': True,
                'technical_accuracy_validator': True,
                'completeness_assessor': True,
                'clarity_assessor': True
            },
            'thresholds': self.thresholds,
            'capabilities': {
                'semantic_relevance_analysis': True,
                'claim_verification': True,
                'technical_validation': True,
                'unit_checking': True,
                'range_validation': True,
                'contradiction_detection': True,
                'completeness_assessment': True,
                'clarity_evaluation': True
            },
            'supported_domains': ['reservoir_engineering', 'petroleum_engineering']
        }


# Global self-critique system instance
_global_self_critique = None

def get_self_critique_system(embedder_model: Optional[SentenceTransformer] = None) -> SelfCritiqueSystem:
    """
    Get or create global self-critique system instance
    """
    global _global_self_critique
    if _global_self_critique is None:
        _global_self_critique = SelfCritiqueSystem(embedder_model)
    return _global_self_critique