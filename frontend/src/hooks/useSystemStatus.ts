import { useState, useEffect } from 'react';
import { apiService } from '../services/api';
import { useAppStore } from '../stores/appStore';

export const useSystemStatus = (refreshInterval: number = 30000) => {
  const { systemStatus, setSystemStatus } = useAppStore();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Erro ao carregar status';
      setError(errorMessage);
      console.error('Failed to fetch system status:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();

    const interval = setInterval(fetchStatus, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  return {
    systemStatus,
    isLoading,
    error,
    refetch: fetchStatus
  };
};