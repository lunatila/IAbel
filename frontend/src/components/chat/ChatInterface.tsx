import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Loader2, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { useAppStore } from '../../stores/appStore';
import { apiService } from '../../services/api';
import { Message, Source } from '../../types';
import ChatMessage from './ChatMessage';
import WelcomeScreen from './WelcomeScreen';
import Button from '../ui/Button';
import Input from '../ui/Input';
import Badge from '../ui/Badge';
import toast from 'react-hot-toast';
import chatIcon from '../../images/IAbel_cigarro.png';

// Helper function to clean preview text
const cleanPreviewText = (text: string): string => {
  if (!text) return '';
  
  // Fix common PDF extraction issues
  let cleaned = text
    // Add space between lowercase and uppercase letters
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    // Add space between letters and numbers
    .replace(/([a-zA-Z])(\d)/g, '$1 $2')
    .replace(/(\d)([a-zA-Z])/g, '$1 $2')
    // Add space after punctuation if missing
    .replace(/([.!?:;,])([A-Za-z])/g, '$1 $2')
    // Fix common Portuguese word combinations that get stuck together
    .replace(/([a-z])([A-Z][a-z]{2,})/g, '$1 $2')
    // Add space between words that are clearly separate (heuristic)
    .replace(/([a-z]{3,})([A-Z][a-z]{3,})/g, '$1 $2')
    // Fix specific patterns common in Portuguese technical text
    .replace(/([a-z])de([A-Z])/g, '$1 de $2')
    .replace(/([a-z])da([A-Z])/g, '$1 da $2')
    .replace(/([a-z])do([A-Z])/g, '$1 do $2')
    .replace(/([a-z])para([A-Z])/g, '$1 para $2')
    .replace(/([a-z])com([A-Z])/g, '$1 com $2')
    .replace(/([a-z])que([A-Z])/g, '$1 que $2')
    .replace(/([a-z])entre([A-Z])/g, '$1 entre $2')
    .replace(/([a-z])são([A-Z])/g, '$1 são $2')
    // Additional patterns for technical terms
    .replace(/([a-z])volume([A-Z])/g, '$1 volume $2')
    .replace(/([a-z])poros([A-Z])/g, '$1 poros $2')
    .replace(/([a-z])poços([A-Z])/g, '$1 poços $2')
    .replace(/([a-z])parâmetros([A-Z])/g, '$1 parâmetros $2')
    .replace(/([a-z])modelo([A-Z])/g, '$1 modelo $2')
    // Normalize multiple spaces to single space
    .replace(/\s+/g, ' ')
    .trim();
  
  // Truncate if too long
  if (cleaned.length > 200) {
    // Try to break at a word boundary
    const truncated = cleaned.substring(0, 200);
    const lastSpace = truncated.lastIndexOf(' ');
    cleaned = (lastSpace > 150 ? truncated.substring(0, lastSpace) : truncated) + '...';
  }
  
  return cleaned;
};

const ChatInterface: React.FC = () => {
  const {
    messages,
    isLoading,
    connectionStatus,
    conversationId,
    currentStreamingMessage,
    addMessage,
    updateStreamingMessage,
    completeStreamingMessage,
    setLoading,
    setConnectionStatus,
    setConversationId,
    resetChat
  } = useAppStore();

  const [inputValue, setInputValue] = useState('');
  const [ragMode, setRagMode] = useState<'rag_v1' | 'rag_v2' | 'rag_v3'>('rag_v2');
  const [messageQuestions, setMessageQuestions] = useState<Map<string, string>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamingContentRef = useRef<string>('');
  const inputRef = useRef<HTMLInputElement>(null);

  // Handle streaming chunks
  const handleStreamChunk = (
    chunk: any,
    setFinalResponse: (response: any) => void
  ) => {
    switch (chunk.type) {
      case 'start':
        updateStreamingMessage('');
        if (chunk.conversation_id) {
          setConversationId(chunk.conversation_id);
        }
        break;

      case 'context':
        updateStreamingMessage('');
        break;

      case 'search':
        updateStreamingMessage('');
        break;

      case 'search_complete':
        updateStreamingMessage('');
        break;

      case 'compression':
        updateStreamingMessage('');
        break;

      case 'generation_start':
        // Reset content for new generation and prepare for tokens
        streamingContentRef.current = '';
        updateStreamingMessage('');
        break;

      case 'token':
        // Accumulate tokens for streaming effect - use ref to ensure we have latest content
        const token = chunk.content || '';
        streamingContentRef.current = streamingContentRef.current + token;
        console.log('Token received:', token, 'Accumulated:', streamingContentRef.current.length, 'chars');

        // Force immediate UI update
        updateStreamingMessage(streamingContentRef.current);
        break;

      case 'complete':
        // Final response with metadata
        setFinalResponse(chunk);
        updateStreamingMessage('');
        setLoading(false);
        break;

      case 'error':
        updateStreamingMessage('');
        setLoading(false);
        toast.error(chunk.message || 'Erro no processamento');
        break;

      default:
        console.log('Unknown stream chunk type:', chunk.type);
    }
  };

  // Auto scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentStreamingMessage]);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      setConnectionStatus('connecting');
      
      const isOnline = await apiService.healthCheck();
      if (isOnline) {
        setConnectionStatus('connected');
        toast.success('Backend conectado');
      } else {
        setConnectionStatus('error');
        toast.error('Backend não está disponível');
      }
    } catch (error) {
      console.error('Failed to check backend health:', error);
      setConnectionStatus('error');
      toast.error('Falha ao conectar com o backend');
    }
  };



  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      isUser: true,
      timestamp: new Date()
    };

    addMessage(userMessage);
    const originalQuestion = inputValue.trim();
    setInputValue('');
    
    // Save original question for reference
    setMessageQuestions(prev => new Map(prev.set(userMessage.id, originalQuestion)));

    try {
      setLoading(true);
      streamingContentRef.current = '';
      let finalResponse: any = null;
      
      // Clear any previous streaming message
      updateStreamingMessage('');

      // Use new streaming API
      await apiService.sendMessageStream(
        userMessage.content,
        ragMode,
        conversationId || undefined,
        4, // top_k
        true, // include_sources
        (chunk) => {
          console.log('Received chunk:', chunk); // Debug logging
          handleStreamChunk(chunk, (response) => {
            finalResponse = response;
          });
        }
      );

      // Create final AI message
      if (finalResponse) {
        // Map backend source format to frontend format
        const mappedSources = (finalResponse.sources || []).map((backendSource: any) => ({
          source: backendSource.title || backendSource.source || 'Documento sem título',
          page: backendSource.page && backendSource.page !== 'null' && backendSource.page !== null ? backendSource.page : 'N/A',
          similarity: backendSource.similarity || 0,
          preview: cleanPreviewText(backendSource.content || backendSource.preview || ''),
          matched_query: backendSource.matched_query || '',
          priority_section: backendSource.priority_section,
          boost_applied: backendSource.boost_applied,
          fusion_score: backendSource.fusion_score,
          rerank_score: backendSource.rerank_score,
          section_type: backendSource.section_type,
          citation_number: backendSource.citation_number
        }));

        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: finalResponse.answer || streamingContentRef.current,
          isUser: false,
          timestamp: new Date(),
          confidence: finalResponse.confidence || 0,
          sources: mappedSources,
          // Enhanced RAG features
          response_id: finalResponse.response_id,
          enhanced: finalResponse.enhanced || true,
          compression_stats: finalResponse.compression_stats,
          originalQuestion
        };

        completeStreamingMessage(aiMessage);
      } else {
        // Fallback if no final response
        const aiMessage: Message = {
          id: (Date.now() + 1).toString(),
          content: streamingContentRef.current || 'Erro ao gerar resposta',
          isUser: false,
          timestamp: new Date(),
          confidence: 0,
          sources: [],
          originalQuestion
        };

        completeStreamingMessage(aiMessage);
      }

    } catch (error) {
      console.error('Error sending message:', error);
      setLoading(false);
      updateStreamingMessage('');
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Erro ao enviar mensagem: ${error instanceof Error ? error.message : 'Erro desconhecido'}`,
        isUser: false,
        timestamp: new Date()
      };
      addMessage(errorMessage);
      toast.error('Erro ao enviar mensagem');
    }
  };

  const handleSourceClick = (source: Source) => {
    toast.success(`Fonte: ${source.source} (Página ${source.page})`);
    // TODO: Implement source modal or navigation
  };

  const getConnectionStatusBadge = () => {
    const statusConfig = {
      connected: { variant: 'success' as const, icon: Wifi, text: 'Conectado' },
      connecting: { variant: 'warning' as const, icon: Loader2, text: 'Conectando...' },
      disconnected: { variant: 'default' as const, icon: WifiOff, text: 'Desconectado' },
      error: { variant: 'error' as const, icon: WifiOff, text: 'Erro' }
    };

    const config = statusConfig[connectionStatus];
    const Icon = config.icon;

    return (
      <Badge variant={config.variant} size="sm">
        <Icon size={12} className={connectionStatus === 'connecting' ? 'animate-spin' : ''} />
        <span className="ml-1">{config.text}</span>
      </Badge>
    );
  };

  const handleSuggestedQuestion = (question: string) => {
    setInputValue(question);
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] bg-white dark:bg-gray-900 rounded-2xl shadow-xl overflow-hidden">
      {/* Header - Minimalist like ChatGPT */}
      {messages.length > 0 && (
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              IAbel
            </h2>
            {getConnectionStatusBadge()}
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={checkBackendHealth}
              disabled={connectionStatus === 'connecting'}
              icon={<RefreshCw size={14} />}
            >
              Reconectar
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={resetChat}
            >
              Nova Conversa
            </Button>
          </div>
        </div>
      )}

      {messages.length === 0 && !currentStreamingMessage ? (
        /* Welcome Screen with centered input */
        <div className="flex-1 flex flex-col items-center justify-center px-4">
          <WelcomeScreen
            onSuggestedQuestion={handleSuggestedQuestion}
            ragMode={ragMode}
            onRagModeChange={setRagMode}
          />

          {/* Centered Input */}
          <div className="w-full max-w-3xl mt-8">
            <form onSubmit={handleSendMessage} className="relative">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Mensagem para IAbel..."
                disabled={isLoading}
                className="pr-12 py-4 text-base rounded-3xl bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700
                           focus:bg-white dark:focus:bg-gray-800 transition-colors"
              />

              <Button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                isLoading={isLoading}
                className="absolute right-2 top-1/2 -translate-y-1/2 !rounded-full w-10 h-10 p-0 flex items-center justify-center"
                icon={!isLoading && <Send size={18} />}
              />
            </form>
          </div>
        </div>
      ) : (
        <>
          {/* Messages Area - Full width like ChatGPT */}
          <div className="flex-1 overflow-y-auto">
            <AnimatePresence>
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onSourceClick={handleSourceClick}
                  originalQuestion={messageQuestions.get(message.id)}
                  ragMode={ragMode}
                />
              ))}
            </AnimatePresence>

            {/* Streaming Message - ChatGPT style */}
            {currentStreamingMessage && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full bg-gray-50/50 dark:bg-gray-800/30"
              >
                <div className="max-w-4xl mx-auto px-4 py-6 flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/30 flex items-center justify-center relative">
                      <img
                        src={chatIcon}
                        alt="IAbel"
                        className="w-6 h-6 object-cover rounded-full"
                      />
                      <Loader2 size={12} className="animate-spin text-white absolute -top-1 -right-1" />
                    </div>
                  </div>

                  <div className="flex-1 space-y-3 min-w-0">
                    <div className="text-sm font-semibold text-gray-900 dark:text-white">
                      IAbel
                    </div>
                    <div className="text-[15px] leading-7 text-gray-800 dark:text-gray-200">
                      {currentStreamingMessage.split('\n').map((line, index) => {
                        if (line.trim() === '') return <br key={index} />;

                        const formattedLine = line
                          .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
                          .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
                          .replace(/•/g, '•');

                        return (
                          <p
                            key={index}
                            className="mb-3 last:mb-0"
                            dangerouslySetInnerHTML={{ __html: formattedLine }}
                          />
                        );
                      })}
                      <span className="inline-block w-2 h-5 ml-1 bg-blue-500 animate-pulse"></span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area - ChatGPT style bottom bar */}
          <div className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
            <div className="max-w-4xl mx-auto px-4 py-4">
              {/* RAG Mode Selector - Only show when there are messages */}
              {messages.length > 0 && (
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Modo:</span>
                  <div className="flex gap-1">
                    <Button
                      variant={ragMode === 'rag_v1' ? 'primary' : 'ghost'}
                      size="sm"
                      onClick={() => setRagMode('rag_v1')}
                      disabled={isLoading}
                      className="text-xs px-3 py-1"
                    >
                      v1
                    </Button>
                    <Button
                      variant={ragMode === 'rag_v2' ? 'primary' : 'ghost'}
                      size="sm"
                      onClick={() => setRagMode('rag_v2')}
                      disabled={isLoading}
                      className="text-xs px-3 py-1"
                    >
                      v2
                    </Button>
                    <Button
                      variant={ragMode === 'rag_v3' ? 'primary' : 'ghost'}
                      size="sm"
                      onClick={() => setRagMode('rag_v3')}
                      disabled={isLoading}
                      className="text-xs px-3 py-1"
                    >
                      v3
                    </Button>
                  </div>
                </div>
              )}

              {/* Input container - centered like ChatGPT */}
              <form onSubmit={handleSendMessage} className="relative">
                <Input
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Mensagem para IAbel..."
                  disabled={isLoading}
                  className="pr-12 py-4 text-base rounded-3xl bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700
                             focus:bg-white dark:focus:bg-gray-800 transition-colors"
                />

                <Button
                  type="submit"
                  disabled={isLoading || !inputValue.trim()}
                  isLoading={isLoading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 !rounded-full w-10 h-10 p-0 flex items-center justify-center"
                  icon={!isLoading && <Send size={18} />}
                />
              </form>

            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ChatInterface;