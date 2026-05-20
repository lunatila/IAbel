// Core API Types
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  message?: string;
}

// Chat Types
export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  sources?: Source[];
  confidence?: number;
  cached?: boolean;
  // Enhanced RAG features
  response_id?: string;
  enhanced?: boolean;
  generation_mode?: string;
  enhancements_used?: EnhancementsUsed;
  compression_stats?: CompressionStats;
  citation_stats?: CitationStats;
  feedback?: UserFeedback;
  originalQuestion?: string;
}

export interface Source {
  source: string;
  page: number | string;
  similarity: number;
  preview: string;
  matched_query: string;
  priority_section?: string;
  boost_applied?: number;
  // Enhanced RAG features
  fusion_score?: number;
  rerank_score?: number;
  section_type?: string;
  // Academic citation features
  citation_number?: number;
}

export interface ChatResponse {
  answer: string;
  confidence: number;
  conversation_id: string;
  total_sources: number;
  sources?: Source[];
  cached?: boolean;
  timestamp: string;
  // Enhanced RAG features
  enhanced?: boolean;
  generation_mode?: string;
  response_id?: string;
  enhancements_used?: EnhancementsUsed;
  compression_stats?: CompressionStats;
  citation_stats?: CitationStats;
  feedback_adjustments?: FeedbackAdjustments;
}

export interface EnhancementsUsed {
  semantic_chunking: boolean;
  rag_fusion: boolean;
  reranking: boolean;
  context_enhancement: boolean;
  context_compression: boolean;
  citation_tracking: boolean;
  feedback_learning: boolean;
}

export interface CompressionStats {
  compression_ratio: number;
  relevance_score: number;
  bytes_saved: number;
}

export interface CitationStats {
  coverage: number;
  source_diversity: number;
  cited_segments: number;
  uncited_segments: number;
}

export interface FeedbackAdjustments {
  applied: boolean;
  confidence: number;
  adjustments_count: number;
}

// Feedback System Types
export interface UserFeedback {
  feedback_id?: string;
  rating: number; // 1-5 scale
  feedback_type: 'thumbs_up' | 'thumbs_down' | 'rating' | 'detailed';
  feedback_text?: string;
  aspects?: FeedbackAspects;
  source_quality?: number;
  citation_quality?: number;
  timestamp?: string;
}

export interface FeedbackAspects {
  accuracy: number;
  relevance: number;
  completeness: number;
  clarity: number;
}

export interface FeedbackRequest {
  response_id: string;
  user_id: string;
  question: string;
  response_text: string;
  rating: number;
  feedback_type?: string;
  feedback_text?: string;
  aspects?: FeedbackAspects;
  source_quality?: number;
  citation_quality?: number;
}

export interface FeedbackStats {
  feedback_enabled: boolean;
  total_feedback: number;
  average_rating: number;
  rating_distribution: Record<number, number>;
  learned_patterns: number;
  average_pattern_confidence: number;
  recent_feedback_count: number;
  success_rate: number;
  error?: string;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'start' | 'search' | 'context' | 'answer_chunk' | 'sources' | 'complete' | 'error';
  message?: string;
  content?: string;
  is_complete?: boolean;
  confidence?: number;
  conversation_id?: string;
  sources?: Source[];
  final_answer?: string;
  error?: string;
  timestamp: string;
}

// Search Types
export interface SearchQuery {
  query: string;
  top_k?: number;
  similarity_threshold?: number;
}

export interface SearchResult {
  content: string;
  source: string;
  page: number | string;
  similarity: number;
  priority_section: string;
  matched_query: string;
  boost_applied: number;
}

export interface SearchResponse {
  results: SearchResult[];
  total_found: number;
  query_processed: string;
  cached?: boolean;
  timestamp: string;
}

// System Status Types
export interface SystemStatus {
  service: string;
  pdf_directory: string;
  vectorstore_path: string;
  active_conversations: number;
  cache: CacheStats;
  timestamp: string;
  capabilities: {
    multilingual_search: boolean;
    context_enhancement: boolean;
    priority_boosting: boolean;
    streaming_responses: boolean;
    acronym_definitions: boolean;
    redis_caching: boolean;
  };
  vector_store: {
    total_documents: number;
    collection_name: string;
    persist_directory: string;
  };
  embedder: {
    model: string;
    dimension: number;
    device: string;
  };
  llm: {
    model: string;
    status: 'online' | 'offline';
    base_url: string;
  };
  processor: {
    chunk_size: number;
    chunk_overlap: number;
  };
  error_tracking?: ErrorTracking;
}

export interface CacheStats {
  cache_type: 'Redis' | 'In-Memory';
  connected: boolean;
  used_memory?: string;
  total_keys?: number;
  hits?: number;
  misses?: number;
  memory_usage?: string;
  error?: string;
}

export interface ErrorTracking {
  total_unique_errors: number;
  total_error_occurrences: number;
  most_common_errors: [string, number][];
  recent_errors_count: number;
}

export interface ErrorSummary {
  summary: ErrorTracking;
  recent_errors: RecentError[];
}

export interface RecentError {
  timestamp: string;
  type: string;
  message: string;
  context: Record<string, any>;
}

// PDF Upload Types
export interface UploadResponse {
  message: string;
  chunks_created?: number;
  filename: string;
  timestamp: string;
  success?: boolean;
  error?: string;
}

// Reindex Types
export interface ReindexRequest {
  force_reindex?: boolean;
}

export interface ReindexUpdate {
  type: 'start' | 'progress' | 'complete' | 'error';
  message: string;
  total_files?: number;
  total_documents?: number;
  embedder_model?: string;
  llm_model?: string;
  error?: string;
  timestamp: string;
}

// UI State Types
export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  isConnected: boolean;
  conversationId: string | null;
  currentStreamingMessage: string;
}

export interface AppState {
  darkMode: boolean;
  systemStatus: SystemStatus | null;
  isSystemOnline: boolean;
  lastStatusCheck: Date | null;
}

// Component Props Types
export interface ChatMessageProps {
  message: Message;
  onSourceClick?: (source: Source) => void;
}

export interface SearchResultProps {
  result: SearchResult;
  onContentClick?: (content: string) => void;
}

export interface StatusCardProps {
  title: string;
  value: string | number;
  description?: string;
  status?: 'success' | 'warning' | 'error' | 'info';
  icon?: React.ComponentType<any>;
}

// Utility Types
export type Theme = 'light' | 'dark';
export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected' | 'error';