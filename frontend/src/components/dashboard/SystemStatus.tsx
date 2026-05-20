import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Server, 
  Database, 
  Brain, 
  Zap, 
  FileText, 
  Clock, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Trash2
} from 'lucide-react';
import { apiService } from '../../services/api';
import { useAppStore } from '../../stores/appStore';
import { SystemStatus as SystemStatusType, ErrorSummary } from '../../types';
import Button from '../ui/Button';
import Card from '../ui/Card';
import Badge from '../ui/Badge';
import Spinner from '../ui/Spinner';
import toast from 'react-hot-toast';

interface StatusCardProps {
  title: string;
  value: string | number;
  description?: string;
  status?: 'success' | 'warning' | 'error' | 'info';
  icon: React.ComponentType<any>;
  loading?: boolean;
}

const StatusCard: React.FC<StatusCardProps> = ({
  title,
  value,
  description,
  status = 'info',
  icon: Icon,
  loading = false
}) => {
  const statusColors = {
    success: 'text-green-600 dark:text-green-400',
    warning: 'text-yellow-600 dark:text-yellow-400',
    error: 'text-red-600 dark:text-red-400',
    info: 'text-blue-600 dark:text-blue-400'
  };

  return (
    <Card>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-1">
            <Icon size={16} className={statusColors[status]} />
            <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {title}
            </h3>
          </div>
          
          <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
            {loading ? <Spinner size="sm" /> : value}
          </div>
          
          {description && (
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {description}
            </p>
          )}
        </div>
      </div>
    </Card>
  );
};

const SystemStatus: React.FC = () => {
  const { systemStatus, setSystemStatus } = useAppStore();
  const [errorSummary, setErrorSummary] = useState<ErrorSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    fetchSystemStatus();
    fetchErrorSummary();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      fetchSystemStatus();
      fetchErrorSummary();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const fetchSystemStatus = async () => {
    try {
      setIsLoading(true);
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch system status:', error);
      toast.error('Erro ao carregar status do sistema');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchErrorSummary = async () => {
    try {
      const summary = await apiService.getErrorSummary();
      setErrorSummary(summary);
    } catch (error) {
      console.error('Failed to fetch error summary:', error);
    }
  };

  const handleClearCache = async () => {
    try {
      await apiService.clearCache();
      toast.success('Cache limpo com sucesso');
      fetchSystemStatus(); // Refresh status
    } catch (error) {
      console.error('Failed to clear cache:', error);
      toast.error('Erro ao limpar cache');
    }
  };

  const getLLMStatus = (status: string) => {
    return status === 'online' ? 'success' : 'error';
  };

  const getCacheStatus = (connected: boolean) => {
    return connected ? 'success' : 'warning';
  };

  if (!systemStatus) {
    return (
      <Card className="flex items-center justify-center py-8">
        <div className="flex items-center space-x-3">
          <Spinner size="md" />
          <span className="text-gray-600 dark:text-gray-400">
            Carregando status do sistema...
          </span>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Status do Sistema
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Monitoramento em tempo real do IAbel RAG System
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearCache}
            icon={<Trash2 size={14} />}
          >
            Limpar Cache
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              fetchSystemStatus();
              fetchErrorSummary();
            }}
            isLoading={isLoading}
            icon={!isLoading && <RefreshCw size={14} />}
          >
            Atualizar
          </Button>
        </div>
      </div>

      {/* Last Update */}
      {lastUpdate && (
        <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
          <Clock size={14} />
          <span>
            Última atualização: {lastUpdate.toLocaleTimeString('pt-BR')}
          </span>
        </div>
      )}

      {/* Main Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatusCard
          title="Vector Store"
          value={systemStatus.vector_store.total_documents}
          description="Documentos indexados"
          status="success"
          icon={Database}
          loading={isLoading}
        />

        <StatusCard
          title="LLM Status"
          value={systemStatus.llm.status === 'online' ? 'Online' : 'Offline'}
          description={systemStatus.llm.model}
          status={getLLMStatus(systemStatus.llm.status)}
          icon={Brain}
          loading={isLoading}
        />

        <StatusCard
          title="Cache"
          value={systemStatus.cache.total_keys || 0}
          description={`${systemStatus.cache.cache_type} - ${systemStatus.cache.connected ? 'Conectado' : 'Desconectado'}`}
          status={getCacheStatus(systemStatus.cache.connected)}
          icon={Zap}
          loading={isLoading}
        />

        <StatusCard
          title="Conversas Ativas"
          value={systemStatus.active_conversations}
          description="Sessões de chat ativas"
          status="info"
          icon={TrendingUp}
          loading={isLoading}
        />
      </div>

      {/* Detailed Information */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* System Capabilities */}
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
            <CheckCircle size={18} className="mr-2 text-green-600" />
            Capacidades do Sistema
          </h3>
          
          <div className="space-y-2">
            {Object.entries(systemStatus.capabilities).map(([key, enabled]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300 capitalize">
                  {key.replace(/_/g, ' ')}
                </span>
                <Badge variant={enabled ? 'success' : 'error'} size="sm">
                  {enabled ? 'Ativo' : 'Inativo'}
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        {/* Cache Statistics */}
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
            <Zap size={18} className="mr-2 text-blue-600" />
            Estatísticas do Cache
          </h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">Tipo</span>
              <span className="text-sm font-medium">{systemStatus.cache.cache_type}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-700 dark:text-gray-300">Status</span>
              <Badge variant={systemStatus.cache.connected ? 'success' : 'error'} size="sm">
                {systemStatus.cache.connected ? 'Conectado' : 'Desconectado'}
              </Badge>
            </div>
            
            {systemStatus.cache.total_keys !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Total de Chaves</span>
                <span className="text-sm font-medium">{systemStatus.cache.total_keys}</span>
              </div>
            )}
            
            {systemStatus.cache.used_memory && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Memória Usada</span>
                <span className="text-sm font-medium">{systemStatus.cache.used_memory}</span>
              </div>
            )}
            
            {systemStatus.cache.hits !== undefined && systemStatus.cache.misses !== undefined && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Taxa de Acerto</span>
                <span className="text-sm font-medium">
                  {Math.round((systemStatus.cache.hits / (systemStatus.cache.hits + systemStatus.cache.misses)) * 100)}%
                </span>
              </div>
            )}
          </div>
        </Card>

        {/* Model Information */}
        <Card>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
            <Brain size={18} className="mr-2 text-purple-600" />
            Modelos de IA
          </h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Modelo de Embedding
              </h4>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Modelo</span>
                  <span className="text-xs font-mono">{systemStatus.embedder.model}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Dimensões</span>
                  <span className="text-xs">{systemStatus.embedder.dimension}D</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Dispositivo</span>
                  <Badge variant="info" size="sm">{systemStatus.embedder.device}</Badge>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                Modelo LLM
              </h4>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Modelo</span>
                  <span className="text-xs font-mono">{systemStatus.llm.model}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Status</span>
                  <Badge variant={getLLMStatus(systemStatus.llm.status)} size="sm">
                    {systemStatus.llm.status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 dark:text-gray-400">URL Base</span>
                  <span className="text-xs font-mono">{systemStatus.llm.base_url}</span>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Error Tracking */}
        {errorSummary && (
          <Card>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <AlertTriangle size={18} className="mr-2 text-yellow-600" />
              Rastreamento de Erros
            </h3>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Total de Erros</span>
                <span className="text-sm font-medium">
                  {errorSummary.summary.total_error_occurrences}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Tipos Únicos</span>
                <span className="text-sm font-medium">
                  {errorSummary.summary.total_unique_errors}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-700 dark:text-gray-300">Recentes</span>
                <span className="text-sm font-medium">
                  {errorSummary.summary.recent_errors_count}
                </span>
              </div>

              {errorSummary.summary.most_common_errors.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">
                    Erros Mais Comuns
                  </h4>
                  <div className="space-y-1">
                    {errorSummary.summary.most_common_errors.slice(0, 3).map(([error, count], index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-xs text-gray-600 dark:text-gray-400 truncate flex-1 mr-2">
                          {error}
                        </span>
                        <Badge variant="error" size="sm">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default SystemStatus;