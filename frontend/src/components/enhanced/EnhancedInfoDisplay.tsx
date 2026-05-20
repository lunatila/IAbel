import React, { useState } from 'react';
import { 
  Zap, 
  BookOpen, 
  Brain, 
  Archive, 
  Quote, 
  TrendingUp, 
  ChevronDown, 
  ChevronRight,
  Info
} from 'lucide-react';
import Badge from '../ui/Badge';
import Card from '../ui/Card';
import Button from '../ui/Button';
import { 
  EnhancementsUsed, 
  CompressionStats, 
  CitationStats, 
  FeedbackAdjustments 
} from '../../types';

interface EnhancedInfoDisplayProps {
  enhanced?: boolean;
  generationMode?: string;
  enhancementsUsed?: EnhancementsUsed;
  compressionStats?: CompressionStats;
  citationStats?: CitationStats;
  feedbackAdjustments?: FeedbackAdjustments;
  compact?: boolean;
}

export const EnhancedInfoDisplay: React.FC<EnhancedInfoDisplayProps> = ({
  enhanced = false,
  generationMode,
  enhancementsUsed,
  compressionStats,
  citationStats,
  feedbackAdjustments,
  compact = false
}) => {
  const [expanded, setExpanded] = useState(false);

  if (!enhanced || !enhancementsUsed) {
    return null;
  }

  const getGenerationModeIcon = (mode?: string) => {
    switch (mode) {
      case 'rag_v2':
        return <Zap className="h-3 w-3" />;
      case 'rag_v1':
        return <BookOpen className="h-3 w-3" />;
      default:
        return <Info className="h-3 w-3" />;
    }
  };

  const getGenerationModeColor = (mode?: string) => {
    switch (mode) {
      case 'rag_v2':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'rag_v1':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
    }
  };

  const activeEnhancements = Object.entries(enhancementsUsed)
    .filter(([_, active]) => active)
    .map(([name, _]) => name);

  const enhancementIcons: Record<string, React.ReactNode> = {
    semantic_chunking: <Brain className="h-3 w-3" />,
    rag_fusion: <TrendingUp className="h-3 w-3" />,
    reranking: <Zap className="h-3 w-3" />,
    context_enhancement: <BookOpen className="h-3 w-3" />,
    context_compression: <Archive className="h-3 w-3" />,
    citation_tracking: <Quote className="h-3 w-3" />,
    feedback_learning: <Brain className="h-3 w-3" />
  };

  const enhancementLabels: Record<string, string> = {
    semantic_chunking: 'Chunking Semântico',
    rag_fusion: 'RAG Fusion',
    reranking: 'Re-ranking',
    context_enhancement: 'Contexto Aprimorado',
    context_compression: 'Compressão de Contexto',
    citation_tracking: 'Rastreamento de Citações',
    feedback_learning: 'Aprendizado com Feedback'
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
        <Badge variant="secondary" className={`text-xs ${getGenerationModeColor(generationMode)}`}>
          {getGenerationModeIcon(generationMode)}
          <span className="ml-1">{generationMode?.toUpperCase()}</span>
        </Badge>
        
        {activeEnhancements.length > 0 && (
          <div className="flex items-center gap-1">
            <span className="text-xs">+{activeEnhancements.length} melhorias</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setExpanded(!expanded)}
              className="p-0 h-4 w-4"
            >
              {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <Card className="mt-3 p-3 bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant="secondary" className={getGenerationModeColor(generationMode)}>
            {getGenerationModeIcon(generationMode)}
            <span className="ml-1">Enhanced RAG {generationMode?.replace('rag_', 'v').toUpperCase()}</span>
          </Badge>
          
          {feedbackAdjustments?.applied && (
            <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
              <Brain className="h-3 w-3 mr-1" />
              Ajustes Aplicados
            </Badge>
          )}
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded(!expanded)}
          className="text-blue-600 dark:text-blue-400"
        >
          {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </Button>
      </div>

      {expanded && (
        <div className="mt-3 space-y-3">
          {/* Active Enhancements */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Melhorias Ativas
            </h4>
            <div className="flex flex-wrap gap-1">
              {activeEnhancements.map((enhancement) => (
                <Badge
                  key={enhancement}
                  variant="secondary"
                  className="text-xs bg-blue-100 text-blue-800 dark:bg-blue-800 dark:text-blue-200"
                >
                  {enhancementIcons[enhancement]}
                  <span className="ml-1">{enhancementLabels[enhancement]}</span>
                </Badge>
              ))}
            </div>
          </div>

          {/* Compression Stats */}
          {compressionStats && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Estatísticas de Compressão
              </h4>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="bg-white dark:bg-gray-800 p-2 rounded border">
                  <div className="font-medium">Taxa de Compressão</div>
                  <div className="text-blue-600 dark:text-blue-400">
                    {(compressionStats.compression_ratio * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded border">
                  <div className="font-medium">Relevância</div>
                  <div className="text-green-600 dark:text-green-400">
                    {(compressionStats.relevance_score * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded border">
                  <div className="font-medium">Bytes Economizados</div>
                  <div className="text-purple-600 dark:text-purple-400">
                    {compressionStats.bytes_saved}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Citation Stats */}
          {citationStats && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Estatísticas de Citações
              </h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-white dark:bg-gray-800 p-2 rounded border">
                  <div className="font-medium">Cobertura</div>
                  <div className="text-blue-600 dark:text-blue-400">
                    {(citationStats.coverage * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-white dark:bg-gray-800 p-2 rounded border">
                  <div className="font-medium">Diversidade de Fontes</div>
                  <div className="text-green-600 dark:text-green-400">
                    {citationStats.source_diversity}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Feedback Adjustments */}
          {feedbackAdjustments?.applied && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Ajustes Baseados em Feedback
              </h4>
              <div className="bg-green-50 dark:bg-green-950 p-2 rounded border border-green-200 dark:border-green-800 text-xs">
                <div className="flex justify-between">
                  <span>Confiança dos Ajustes:</span>
                  <span className="font-medium">
                    {(feedbackAdjustments.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Ajustes Aplicados:</span>
                  <span className="font-medium">{feedbackAdjustments.adjustments_count}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};