import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, FileText, Clock, Zap, TrendingUp } from 'lucide-react';
import { apiService } from '../../services/api';
import { SearchQuery, SearchResult } from '../../types';
import Button from '../ui/Button';
import Input from '../ui/Input';
import Card from '../ui/Card';
import Badge from '../ui/Badge';
import Spinner from '../ui/Spinner';
import toast from 'react-hot-toast';

const DocumentSearch: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchStats, setSearchStats] = useState<{
    totalFound: number;
    queryProcessed: string;
    cached: boolean;
    timestamp: string;
  } | null>(null);
  
  // Advanced options
  const [topK, setTopK] = useState(8);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.3);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!query.trim()) {
      toast.error('Digite uma consulta de busca');
      return;
    }

    setIsSearching(true);
    
    try {
      const searchQuery: SearchQuery = {
        query: query.trim(),
        top_k: topK,
        similarity_threshold: similarityThreshold
      };

      const response = await apiService.searchDocuments(searchQuery);
      
      setResults(response.results);
      setSearchStats({
        totalFound: response.total_found,
        queryProcessed: response.query_processed,
        cached: response.cached || false,
        timestamp: response.timestamp
      });

      toast.success(`Encontrados ${response.results.length} documentos relevantes`);
    } catch (error) {
      console.error('Search error:', error);
      toast.error('Erro na busca de documentos');
      setResults([]);
      setSearchStats(null);
    } finally {
      setIsSearching(false);
    }
  };

  const handleResultClick = (result: SearchResult) => {
    toast.success(`Selecionado: ${result.source} (Página ${result.page})`);
    // TODO: Implement result detail modal or copy to chat
  };

  const getPriorityColor = (section: string) => {
    switch (section) {
      case 'abstract': return 'info';
      case 'definition': return 'success';
      case 'equation': return 'warning';
      default: return 'default';
    }
  };

  const getPriorityLabel = (section: string) => {
    switch (section) {
      case 'abstract': return 'Resumo';
      case 'definition': return 'Definição';
      case 'equation': return 'Equação';
      default: return 'Regular';
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Form */}
      <Card>
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="flex space-x-2">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Nome do documento"
              icon={<Search size={16} />}
              disabled={isSearching}
              className="flex-1"
            />
            
            <Button
              type="submit"
              isLoading={isSearching}
              disabled={!query.trim()}
              icon={!isSearching && <Search size={16} />}
            >
              {isSearching ? 'Buscando...' : 'Buscar'}
            </Button>
          </div>

          {/* Advanced Options Toggle */}
          <div className="flex items-center justify-between">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvanced(!showAdvanced)}
              icon={<Filter size={14} />}
            >
              Opções Avançadas
            </Button>

            {searchStats && (
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Clock size={14} />
                <span>
                  {new Date(searchStats.timestamp).toLocaleTimeString('pt-BR')}
                </span>
                {searchStats.cached && (
                  <>
                    <Zap size={14} />
                    <span>Cache</span>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Advanced Options Panel */}
          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-3 pt-3 border-t border-gray-200 dark:border-gray-700"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Número de Resultados
                    </label>
                    <Input
                      type="number"
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      min={1}
                      max={20}
                      className="w-full"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Limite de Similaridade
                    </label>
                    <Input
                      type="number"
                      value={similarityThreshold}
                      onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
                      min={0}
                      max={1}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </form>
      </Card>

      {/* Search Stats */}
      {searchStats && (
        <Card className="bg-blue-50 dark:bg-blue-900/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <TrendingUp size={16} className="text-blue-600" />
              <span className="text-sm font-medium text-blue-900 dark:text-blue-100">
                {searchStats.totalFound} documentos encontrados para "{searchStats.queryProcessed}"
              </span>
            </div>
            
            {searchStats.cached && (
              <Badge variant="info" size="sm">
                <Zap size={12} />
                <span className="ml-1">Resultado em Cache</span>
              </Badge>
            )}
          </div>
        </Card>
      )}

      {/* Loading State */}
      {isSearching && (
        <Card className="flex items-center justify-center py-8">
          <div className="flex items-center space-x-3">
            <Spinner size="md" />
            <span className="text-gray-600 dark:text-gray-400">
              Buscando documentos relevantes...
            </span>
          </div>
        </Card>
      )}

      {/* Search Results */}
      <AnimatePresence>
        {results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Resultados da Busca
            </h3>
            
            <div className="space-y-3">
              {results.map((result, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card 
                    className="hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => handleResultClick(result)}
                  >
                    <div className="space-y-3">
                      {/* Header */}
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-2 flex-1 min-w-0">
                          <FileText size={16} className="text-gray-500 flex-shrink-0" />
                          <div className="min-w-0 flex-1">
                            <h4 className="font-medium text-gray-900 dark:text-white truncate">
                              {result.source}
                            </h4>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              Página {result.page} • {Math.round(result.similarity * 100)}% similar
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2 flex-shrink-0">
                          <Badge
                            variant={getPriorityColor(result.priority_section)}
                            size="sm"
                          >
                            {getPriorityLabel(result.priority_section)}
                          </Badge>
                          
                          {result.boost_applied > 1 && (
                            <Badge variant="success" size="sm">
                              +{Math.round((result.boost_applied - 1) * 100)}%
                            </Badge>
                          )}
                        </div>
                      </div>

                      {/* Content Preview */}
                      <div className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3">
                        {result.content}
                      </div>

                      {/* Footer */}
                      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                        <span>Query: "{result.matched_query}"</span>
                        <span>Relevância: {Math.round(result.similarity * 1000) / 10}/100</span>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty State */}
      {!isSearching && results.length === 0 && searchStats && (
        <Card className="text-center py-8">
          <FileText size={48} className="mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Nenhum documento encontrado
          </h3>
          <p className="text-gray-500 dark:text-gray-400">
            Tente ajustar sua consulta ou reduzir o limite de similaridade
          </p>
        </Card>
      )}
    </div>
  );
};

export default DocumentSearch;