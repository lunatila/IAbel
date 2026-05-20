import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  ChatResponse, 
  SearchResponse, 
  SearchQuery, 
  SystemStatus, 
  UploadResponse,
  ErrorSummary,
  ReindexUpdate,
  WebSocketMessage,
  FeedbackRequest,
  FeedbackStats
} from '../types';

class ApiService {
  private api: AxiosInstance;
  private baseURL: string;
  private websocket: WebSocket | null = null;

  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    
    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for better error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        
        if (error.response?.status === 500) {
          throw new Error('Erro interno do servidor. Verifique se o backend está funcionando.');
        } else if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
          throw new Error('Não foi possível conectar ao servidor. Verifique se está rodando.');
        }
        
        throw error;
      }
    );
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      await this.api.get('/health/');
      return true;
    } catch {
      return false;
    }
  }

  // Chat methods
  async sendMessage(
    message: string,
    conversationId?: string,
    topK?: number,
    includeSources?: boolean
  ): Promise<ChatResponse> {
    const response: AxiosResponse<ChatResponse> = await this.api.post('/chat/', {
      message,
      conversation_id: conversationId,
      top_k: topK || 6,
      include_sources: includeSources !== false,
    });
    
    return response.data;
  }

  // WebSocket chat streaming
  connectWebSocket(
    onMessage: (message: WebSocketMessage) => void,
    onError?: (error: Event) => void,
    onClose?: (event: CloseEvent) => void
  ): WebSocket {
    const wsUrl = this.baseURL.replace('http', 'ws') + '/chat/stream';
    
    if (this.websocket) {
      this.websocket.close();
    }

    this.websocket = new WebSocket(wsUrl);

    this.websocket.onopen = () => {
      console.log('WebSocket connected');
    };

    this.websocket.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      onError?.(error);
    };

    this.websocket.onclose = (event) => {
      console.log('WebSocket closed:', event);
      onClose?.(event);
    };

    return this.websocket;
  }

  sendWebSocketMessage(message: {
    message: string;
    conversation_id?: string;
    top_k?: number;
  }): void {
    if (this.websocket?.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    } else {
      throw new Error('WebSocket não está conectado');
    }
  }

  closeWebSocket(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  // Search methods
  async searchDocuments(query: SearchQuery): Promise<SearchResponse> {
    const response: AxiosResponse<SearchResponse> = await this.api.post('/search/', query);
    return response.data;
  }

  // System status methods
  async getSystemStatus(): Promise<SystemStatus> {
    const response: AxiosResponse<SystemStatus> = await this.api.get('/status/');
    return response.data;
  }

  async getErrorSummary(): Promise<ErrorSummary> {
    const response: AxiosResponse<ErrorSummary> = await this.api.get('/errors/');
    return response.data;
  }

  // PDF upload methods
  async uploadPDF(file: File, onProgress?: (progress: number) => void): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response: AxiosResponse<UploadResponse> = await this.api.post('/upload-pdf/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  // Reindex methods
  async reindexDocuments(
    forceReindex: boolean = false,
    onUpdate?: (update: ReindexUpdate) => void
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/reindex/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ force_reindex: forceReindex }),
    });

    if (!response.body) {
      throw new Error('No response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const update: ReindexUpdate = JSON.parse(line.slice(6));
            onUpdate?.(update);
          } catch (error) {
            console.error('Failed to parse reindex update:', error);
          }
        }
      }
    }
  }

  // Cache management methods
  async clearCache(): Promise<{ message: string }> {
    const response = await this.api.delete('/cache/');
    return response.data;
  }

  async clearCachePattern(pattern: string): Promise<{ message: string }> {
    const response = await this.api.delete(`/cache/${pattern}`);
    return response.data;
  }

  // Enhanced RAG methods
  async sendMessageWithMode(
    message: string,
    mode: 'rag_v1' | 'rag_v2' | 'rag_v3' | 'lora_only' | 'hybrid' | 'adaptive',
    conversationId?: string,
    topK?: number,
    includeSources?: boolean
  ): Promise<ChatResponse> {
    const response: AxiosResponse<ChatResponse> = await this.api.post('/chat/enhanced/', {
      message,
      mode,
      conversation_id: conversationId,
      top_k: topK || 6,
      include_sources: includeSources !== false,
    });
    
    return response.data;
  }

  // Feedback methods
  async submitFeedback(feedback: FeedbackRequest): Promise<{ feedback_id: string; message: string }> {
    const response = await this.api.post('/feedback/', feedback);
    return response.data;
  }

  async getFeedbackStats(): Promise<FeedbackStats> {
    const response: AxiosResponse<FeedbackStats> = await this.api.get('/feedback/stats/');
    return response.data;
  }

  // Quick feedback methods
  async submitQuickFeedback(
    responseId: string,
    rating: number,
    feedbackType: 'thumbs_up' | 'thumbs_down' | 'rating',
    question: string,
    responseText: string,
    userId: string = 'anonymous'
  ): Promise<{ feedback_id: string; message: string }> {
    const feedback: FeedbackRequest = {
      response_id: responseId,
      user_id: userId,
      question,
      response_text: responseText,
      rating,
      feedback_type: feedbackType,
    };
    
    return this.submitFeedback(feedback);
  }

  async sendMessageStream(
    message: string,
    mode: 'rag_v1' | 'rag_v2' | 'rag_v3' | 'lora_only' | 'hybrid' | 'adaptive',
    conversationId?: string,
    topK?: number,
    includeSources?: boolean,
    onChunk?: (chunk: any) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${this.baseURL}/chat/stream/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          mode,
          conversation_id: conversationId,
          top_k: topK || 4,
          include_sources: includeSources !== false,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No reader available');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'end') {
                return;
              }
              
              if (onChunk) {
                onChunk(data);
              }
            } catch (e) {
              console.warn('Failed to parse streaming data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();