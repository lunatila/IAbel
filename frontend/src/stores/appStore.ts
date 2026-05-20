import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Message, SystemStatus, Theme, ConnectionStatus } from '../types';

interface AppState {
  // Theme
  theme: Theme;
  toggleTheme: () => void;
  
  // Chat
  messages: Message[];
  isLoading: boolean;
  connectionStatus: ConnectionStatus;
  conversationId: string | null;
  currentStreamingMessage: string;
  
  // System
  systemStatus: SystemStatus | null;
  lastStatusCheck: Date | null;
  
  // Actions
  addMessage: (message: Message) => void;
  updateStreamingMessage: (content: string) => void;
  completeStreamingMessage: (finalMessage: Message) => void;
  setLoading: (loading: boolean) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setConversationId: (id: string) => void;
  setSystemStatus: (status: SystemStatus) => void;
  clearMessages: () => void;
  resetChat: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Initial state
      theme: 'light',
      messages: [],
      isLoading: false,
      connectionStatus: 'disconnected',
      conversationId: crypto.randomUUID(),
      currentStreamingMessage: '',
      systemStatus: null,
      lastStatusCheck: null,

      // Theme actions
      toggleTheme: () =>
        set((state) => ({
          theme: state.theme === 'light' ? 'dark' : 'light',
        })),

      // Chat actions
      addMessage: (message) =>
        set((state) => ({
          messages: [...state.messages, message],
        })),

      updateStreamingMessage: (content) =>
        set({ currentStreamingMessage: content }),

      completeStreamingMessage: (finalMessage) =>
        set((state) => ({
          messages: [...state.messages, finalMessage],
          currentStreamingMessage: '',
          isLoading: false,
        })),

      setLoading: (loading) => set({ isLoading: loading }),

      setConnectionStatus: (status) => set({ connectionStatus: status }),

      setConversationId: (id) => set({ conversationId: id }),

      // System actions
      setSystemStatus: (status) =>
        set({
          systemStatus: status,
          lastStatusCheck: new Date(),
        }),

      // Utility actions
      clearMessages: () =>
        set({
          messages: [],
        }),

      resetChat: () =>
        set({
          messages: [],
          conversationId: crypto.randomUUID(), // Generate new conversation ID
          currentStreamingMessage: '',
          isLoading: false,
        }),
    }),
    {
      name: 'iabel-app-store',
      partialize: (state) => ({
        theme: state.theme,
        conversationId: state.conversationId,
      }),
    }
  )
);