import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { User, Clock, Zap } from 'lucide-react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { Message, Source } from '../../types';
import Badge from '../ui/Badge';
import { FeedbackRating } from '../feedback/FeedbackRating';
import { EnhancedInfoDisplay } from '../enhanced/EnhancedInfoDisplay';
import { CitationDisplay } from '../enhanced/CitationDisplay';
import { apiService } from '../../services/api';
import chatIcon from '../../images/IAbel_cigarro.png';

function renderMessageContent(content: string): React.ReactNode {
  const nodes: React.ReactNode[] = [];
  let key = 0;

  // Split on $$...$$ for block math
  const blockSegments = content.split(/(\$\$[\s\S]*?\$\$)/g);

  for (const blockSeg of blockSegments) {
    if (blockSeg.startsWith('$$') && blockSeg.endsWith('$$') && blockSeg.length > 4) {
      const math = blockSeg.slice(2, -2).trim();
      try {
        nodes.push(<div key={key++} className="my-3 overflow-x-auto"><BlockMath math={math} /></div>);
      } catch {
        nodes.push(<code key={key++} className="text-sm bg-gray-100 dark:bg-gray-800 px-1 rounded">{blockSeg}</code>);
      }
      continue;
    }

    // Split on $...$ for inline math (no newlines inside)
    const inlineSegments = blockSeg.split(/(\$[^$\n]+?\$)/g);

    for (const inlineSeg of inlineSegments) {
      if (inlineSeg.startsWith('$') && inlineSeg.endsWith('$') && inlineSeg.length > 2) {
        const math = inlineSeg.slice(1, -1).trim();
        try {
          nodes.push(<InlineMath key={key++} math={math} />);
        } catch {
          nodes.push(<code key={key++} className="text-sm bg-gray-100 dark:bg-gray-800 px-1 rounded">{inlineSeg}</code>);
        }
        continue;
      }

      // Plain text — apply bold/italic and line breaks
      const lines = inlineSeg.split('\n');
      lines.forEach((line, lineIdx) => {
        if (lineIdx > 0) nodes.push(<br key={key++} />);
        if (!line.trim()) return;
        const html = line
          .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
          .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>');
        nodes.push(<span key={key++} dangerouslySetInnerHTML={{ __html: html }} />);
      });
    }
  }

  return <>{nodes}</>;
}

interface ChatMessageProps {
  message: Message;
  onSourceClick?: (source: Source) => void;
  originalQuestion?: string;
  ragMode?: 'rag_v1' | 'rag_v2' | 'rag_v3';
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, onSourceClick, originalQuestion, ragMode }) => {
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('pt-BR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConfidenceColor = (confidence?: number) => {
    if (!confidence) return 'default';
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const formatConfidence = (confidence?: number) => {
    if (!confidence) return null;
    return `${Math.round(confidence * 100)}%`;
  };

  const handleFeedbackSubmit = async (rating: number, feedbackType: 'thumbs_up' | 'thumbs_down' | 'rating') => {
    if (!message.response_id || !originalQuestion) {
      console.warn('Missing response_id or original question for feedback');
      return;
    }

    try {
      await apiService.submitQuickFeedback(
        message.response_id,
        rating,
        feedbackType,
        originalQuestion,
        message.content
      );
      setFeedbackSubmitted(true);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      throw error;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`group w-full ${message.isUser ? 'bg-transparent' : 'bg-gray-50/50 dark:bg-gray-800/30'}`}
    >
      <div className={`max-w-4xl mx-auto px-4 py-6 flex gap-4 ${message.isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className="flex-shrink-0">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            message.isUser
              ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/30'
              : 'bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/30'
          }`}>
            {message.isUser ? (
              <User size={16} strokeWidth={2.5} />
            ) : (
              <img
                src={chatIcon}
                alt="IAbel"
                className="w-6 h-6 object-cover rounded-full"
              />
            )}
          </div>
        </div>

        {/* Message Content */}
        <div className="flex-1 space-y-3 min-w-0">
          {/* User label */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {message.isUser ? 'Você' : 'IAbel'}
            </span>
            {!message.isUser && message.confidence && (
              <Badge
                variant={getConfidenceColor(message.confidence)}
                size="sm"
              >
                {formatConfidence(message.confidence)}
              </Badge>
            )}
          </div>

          {/* Message text - Clean ChatGPT-like style */}
          <div className={`text-[15px] leading-7 ${
            message.isUser
              ? 'text-gray-900 dark:text-gray-100'
              : 'text-gray-800 dark:text-gray-200'
          }`}>
            {renderMessageContent(message.content)}
          </div>

          {/* Metadata - Subtle and clean */}
          {!message.isUser && (
            <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400 pt-1">
              <div className="flex items-center gap-1.5">
                <Clock size={12} />
                <span>{formatTime(message.timestamp)}</span>
              </div>
              {message.cached && (
                <div className="flex items-center gap-1.5">
                  <Zap size={12} className="text-yellow-500" />
                  <span>Cached</span>
                </div>
              )}
            </div>
          )}

          {/* Enhanced RAG Information */}
          {!message.isUser && message.enhanced && (
            <EnhancedInfoDisplay
              enhanced={message.enhanced}
              generationMode={message.generation_mode}
              enhancementsUsed={message.enhancements_used}
              compressionStats={message.compression_stats}
              citationStats={message.citation_stats}
              compact={false}
            />
          )}

          {/* Enhanced Sources Display - Hide for RAG v3 */}
          {!message.isUser && message.sources && message.sources.length > 0 && ragMode !== 'rag_v3' && (
            <CitationDisplay
              sources={message.sources}
              enhanced={message.enhanced}
              onSourceClick={onSourceClick}
              maxSources={5}
            />
          )}

          {/* Feedback System */}
          {!message.isUser && message.response_id && originalQuestion && (
            <div className="mt-3">
              <FeedbackRating
                onRatingSubmit={handleFeedbackSubmit}
                disabled={feedbackSubmitted}
                compact={true}
              />
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default ChatMessage;