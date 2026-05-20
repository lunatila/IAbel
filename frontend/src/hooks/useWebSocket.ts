import { useState, useEffect, useRef, useCallback } from 'react';
import { apiService } from '../services/api';
import { WebSocketMessage, ConnectionStatus } from '../types';

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onError?: (error: Event) => void;
  onClose?: (event: CloseEvent) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

export const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const {
    onMessage,
    onError,
    onClose,
    reconnectAttempts = 5,
    reconnectInterval = 3000
  } = options;

  const handleMessage = useCallback((message: WebSocketMessage) => {
    setLastMessage(message);
    onMessage?.(message);
  }, [onMessage]);

  const handleError = useCallback((error: Event) => {
    console.error('WebSocket error:', error);
    setConnectionStatus('error');
    onError?.(error);
  }, [onError]);

  const handleClose = useCallback((event: CloseEvent) => {
    console.log('WebSocket closed:', event);
    setConnectionStatus('disconnected');
    websocketRef.current = null;
    
    onClose?.(event);

    // Auto-reconnect logic
    if (event.code !== 1000 && reconnectAttemptsRef.current < reconnectAttempts) {
      reconnectAttemptsRef.current += 1;
      
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`Attempting to reconnect... (${reconnectAttemptsRef.current}/${reconnectAttempts})`);
        connect();
      }, reconnectInterval);
    }
  }, [onClose, reconnectAttempts, reconnectInterval]);

  const connect = useCallback(async () => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return websocketRef.current;
    }

    try {
      setConnectionStatus('connecting');
      
      // Check if backend is available
      const isHealthy = await apiService.healthCheck();
      if (!isHealthy) {
        throw new Error('Backend não está disponível');
      }

      const ws = apiService.connectWebSocket(
        handleMessage,
        handleError,
        handleClose
      );

      websocketRef.current = ws;
      
      // Wait for connection to open
      return new Promise<WebSocket>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 10000);

        ws.onopen = () => {
          clearTimeout(timeout);
          setConnectionStatus('connected');
          reconnectAttemptsRef.current = 0; // Reset attempts on successful connection
          resolve(ws);
        };

        ws.onerror = (error) => {
          clearTimeout(timeout);
          reject(error);
        };
      });
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('error');
      throw error;
    }
  }, [handleMessage, handleError, handleClose]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (websocketRef.current) {
      websocketRef.current.close(1000, 'Manual disconnect');
      websocketRef.current = null;
    }

    setConnectionStatus('disconnected');
    reconnectAttemptsRef.current = 0;
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket is not connected');
      return false;
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect().catch(console.error);

    return () => {
      disconnect();
    };
  }, []);

  return {
    connectionStatus,
    lastMessage,
    connect,
    disconnect,
    sendMessage,
    isConnected: connectionStatus === 'connected',
    isConnecting: connectionStatus === 'connecting'
  };
};