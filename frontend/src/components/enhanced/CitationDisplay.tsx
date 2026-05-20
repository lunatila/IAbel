import React from 'react';
import { FileText, Quote, ExternalLink, TrendingUp, Star } from 'lucide-react';
import Badge from '../ui/Badge';
import Button from '../ui/Button';
import { Source } from '../../types';

// Helper function to format source titles
const formatSourceTitle = (title: string): string => {
  if (!title) return 'Documento sem título';
  
  // If it looks like a filename, clean it up
  if (title.includes('.pdf') || title.includes('_compressed')) {
    // Remove file extensions and common suffixes
    let cleaned = title
      .replace(/\.pdf$/i, '')
      .replace(/_compressed$/i, '')
      .replace(/[-_]/g, ' ')
      .trim();
    
    // Capitalize first letter of each word
    cleaned = cleaned.replace(/\b\w/g, l => l.toUpperCase());
    
    return cleaned;
  }
  
  return title;
};

// Helper function to format preview text
const formatPreviewText = (text: string): string => {
  if (!text) return 'Sem prévia disponível';
  
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

interface CitationDisplayProps {
  sources: Source[];
  enhanced?: boolean;
  onSourceClick?: (source: Source) => void;
  maxSources?: number;
}

export const CitationDisplay: React.FC<CitationDisplayProps> = ({
  sources,
  enhanced = false,
  onSourceClick,
  maxSources = 5
}) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  const displaySources = sources.slice(0, maxSources);
  const hasMoreSources = sources.length > maxSources;

  const getSourceIcon = (source: Source) => {
    if (source.section_type) {
      switch (source.section_type) {
        case 'abstract':
          return <Quote className="h-3 w-3" />;
        case 'definition':
          return <Star className="h-3 w-3" />;
        case 'equation':
          return <TrendingUp className="h-3 w-3" />;
        default:
          return <FileText className="h-3 w-3" />;
      }
    }
    return <FileText className="h-3 w-3" />;
  };

  const getSourceTypeColor = (sectionType?: string) => {
    switch (sectionType) {
      case 'abstract':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      case 'definition':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'equation':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
    }
  };

  const getSectionTypeLabel = (sectionType?: string) => {
    switch (sectionType) {
      case 'abstract':
        return 'Resumo';
      case 'definition':
        return 'Definição';
      case 'equation':
        return 'Equação';
      default:
        return 'Conteúdo';
    }
  };

  return (
    <div className="mt-4 space-y-2">
      <div className="flex items-center gap-2">
        <Quote className="h-4 w-4 text-gray-500 dark:text-gray-400" />
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Fontes {/* enhanced && '(Enhanced)'*/}
        </h4>
        {hasMoreSources && (
          <Badge variant="secondary" className="text-xs">
            +{sources.length - maxSources} mais
          </Badge>
        )}
      </div>

      <div className="grid gap-2">
        {displaySources.map((source, index) => (
          <div
            key={index}
            className="flex items-start gap-2 p-2 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors text-sm"
          >
            <div className="flex-shrink-0 mt-1">
              {getSourceIcon(source)}
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  {source.citation_number && (
                    <Badge variant="primary" className="text-xs font-bold">
                      [{source.citation_number}]
                    </Badge>
                  )}
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                    {formatSourceTitle(source.source)}
                  </span>
                  
                  {source.page && (
                    <Badge variant="secondary" className="text-xs">
                      p. {source.page}
                    </Badge>
                  )}
                  
                  {enhanced && source.section_type && (
                    <Badge 
                      variant="secondary" 
                      className={`text-xs ${getSourceTypeColor(source.section_type)}`}
                    >
                      {getSectionTypeLabel(source.section_type)}
                    </Badge>
                  )}
                </div>

                {onSourceClick && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onSourceClick(source)}
                    className="flex-shrink-0 p-1 h-6 w-6"
                  >
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                )}
              </div>

              <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 mb-2">
                {formatPreviewText(source.preview)}
              </p>

              {/* Enhanced Metrics */}
              <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center gap-1">
                  <span>Similaridade:</span>
                  <span className="font-medium">
                    {(source.similarity * 100).toFixed(1)}%
                  </span>
                </div>

                {enhanced && source.fusion_score && (
                  <div className="flex items-center gap-1">
                    <span>Fusion:</span>
                    <span className="font-medium text-blue-600 dark:text-blue-400">
                      {source.fusion_score.toFixed(2)}
                    </span>
                  </div>
                )}

                {enhanced && source.rerank_score && (
                  <div className="flex items-center gap-1">
                    <span>Rerank:</span>
                    <span className="font-medium text-green-600 dark:text-green-400">
                      {source.rerank_score.toFixed(2)}
                    </span>
                  </div>
                )}

                {source.boost_applied && source.boost_applied > 1 && (
                  <div className="flex items-center gap-1">
                    <TrendingUp className="h-3 w-3" />
                    <span className="font-medium text-orange-600 dark:text-orange-400">
                      {source.boost_applied.toFixed(1)}x
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {hasMoreSources && (
        <div className="text-center">
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            Ver todas as {sources.length} fontes
          </Button>
        </div>
      )}
    </div>
  );
};