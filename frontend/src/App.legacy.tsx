import React, { useState, useRef, useEffect } from 'react';
import logo from './images/IAbel_transp.png';
import thinkingLogo from './images/IAbel_cigarro.png';
import upArrow from './images/up_arrow.png';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  confidence?: number;
  sources?: number;
  generationMode?: string;
}

type ChatMode = 'rag_v1' | 'rag_v2' | 'lora_only' | 'hybrid';

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatMode, setChatMode] = useState<ChatMode>('rag_v2');
  const [showModeSelector, setShowModeSelector] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'testing' | 'connected' | 'disconnected'>('testing');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Test backend connectivity on page load
    const testConnection = async () => {
      try {
        console.log('Testing backend connection...');
        const response = await fetch('http://172.28.74.183:8000/health/', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        console.log('Health check response:', response);
        
        if (response.ok) {
          setConnectionStatus('connected');
          console.log('Backend connection: OK');
        } else {
          setConnectionStatus('disconnected');
          console.log('Backend connection: Failed');
        }
      } catch (error) {
        setConnectionStatus('disconnected');
        console.error('Backend connection test failed:', error);
      }
    };
    
    testConnection();
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      const dropdown = target.closest('[data-mode-selector]');
      
      if (showModeSelector && !dropdown) {
        setShowModeSelector(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showModeSelector]);

  const getModeInfo = (mode: ChatMode) => {
    switch (mode) {
      case 'rag_v1':
        return { name: 'RAG v1', description: 'Busca Documental Básica', icon: '📚', color: '#10b981' };
      case 'rag_v2':
        return { name: 'RAG v2', description: 'Busca Avançada + Re-ranking', icon: '🚀', color: '#06b6d4' };
      case 'lora_only':
        return { name: 'LoRA', description: 'Modelo Fine-tuned', icon: '🧠', color: '#8b5cf6' };
      case 'hybrid':
        return { name: 'Hybrid v1.0', description: 'RAG + LoRA', icon: '⚡', color: '#f59e0b' };
    }
  };

  const getModeDisplayName = (mode: ChatMode) => {
    const info = getModeInfo(mode);
    return `${info.icon} ${info.name}`;
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      console.log('Sending request with mode:', chatMode);
      console.log('Message:', inputValue);
      
      const response = await fetch('http://172.28.74.183:8000/chat/hybrid/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputValue,
          conversation_id: null,
          top_k: 5,
          include_sources: true,
          mode: chatMode
        })
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Response data:', data);

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.answer,
        isUser: false,
        timestamp: new Date(),
        confidence: data.confidence,
        sources: data.total_sources,
        generationMode: data.generation_mode
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Erro ao conectar com o backend: ${error instanceof Error ? error.message : 'Erro desconhecido'}. Verifique se o servidor está rodando na porta 8000.`,
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#1a1a1a',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      {/* Header */}
      <div style={{
        backgroundColor: '#44484b',
        borderBottom: '1px solid #555',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <img 
            src={logo} 
            alt="IAbel Logo" 
            style={{
              width: '40px',
              height: '40px',
              borderRadius: '8px',
              objectFit: 'cover'
            }}
          />
          <div>
            <h1 style={{
              margin: 0,
              fontSize: '20px',
              fontWeight: 600,
              color: '#ffffff'
            }}>
              IAbel
            </h1>
            <p style={{
              margin: 0,
              fontSize: '14px',
              color: '#cccccc',
              fontWeight: 400,
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              Especialista em Engenharia de Reservatórios
              <span style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: connectionStatus === 'connected' ? '#10b981' : 
                                connectionStatus === 'disconnected' ? '#ef4444' : '#f59e0b',
                display: 'inline-block'
              }}></span>
              <span style={{ fontSize: '12px', color: '#999' }}>
                {connectionStatus === 'connected' ? 'Online' : 
                 connectionStatus === 'disconnected' ? 'Offline' : 'Testando...'}
              </span>
            </p>
          </div>
        </div>

        {/* Mode Selector */}
        <div style={{ position: 'relative' }} data-mode-selector>
          <button
            onClick={() => {
              console.log('Current mode before toggle:', chatMode);
              setShowModeSelector(!showModeSelector);
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              backgroundColor: getModeInfo(chatMode).color,
              border: 'none',
              borderRadius: '8px',
              color: 'white',
              fontSize: '14px',
              fontWeight: 500,
              cursor: 'pointer',
              transition: 'all 0.2s',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-1px)';
              e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
            }}
          >
            <span>{getModeDisplayName(chatMode)}</span>
            <span style={{ 
              marginLeft: '4px',
              transform: showModeSelector ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s'
            }}>▼</span>
          </button>

          {/* Dropdown Menu */}
          {showModeSelector && (
            <div style={{
              position: 'absolute',
              top: '100%',
              right: 0,
              marginTop: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid #555',
              borderRadius: '8px',
              boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
              zIndex: 1000,
              minWidth: '200px',
              overflow: 'hidden'
            }}>
              {(['rag_v1', 'rag_v2', 'lora_only', 'hybrid'] as ChatMode[]).map((mode) => {
                const info = getModeInfo(mode);
                return (
                  <button
                    key={mode}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      console.log('Selecting mode:', mode);
                      setChatMode(mode);
                      setShowModeSelector(false);
                    }}
                    style={{
                      width: '100%',
                      padding: '12px 16px',
                      backgroundColor: chatMode === mode ? info.color + '20' : 'transparent',
                      border: 'none',
                      borderBottom: mode !== 'hybrid' ? '1px solid #444' : 'none',
                      color: chatMode === mode ? info.color : '#ffffff',
                      fontSize: '14px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}
                    onMouseEnter={(e) => {
                      if (chatMode !== mode) {
                        e.currentTarget.style.backgroundColor = '#3a3a3a';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (chatMode !== mode) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 500 }}>
                        {info.icon} {info.name}
                      </div>
                      <div style={{ 
                        fontSize: '12px', 
                        color: chatMode === mode ? info.color + 'cc' : '#999',
                        marginTop: '2px'
                      }}>
                        {info.description}
                      </div>
                    </div>
                    {chatMode === mode && (
                      <span style={{ color: info.color, fontSize: '16px' }}>✓</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Messages Container */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: messages.length === 0 ? '0' : '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        maxHeight: messages.length === 0 ? 'none' : '70vh'
      }}>
        {messages.length === 0 && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '60vh',
            textAlign: 'center',
            color: '#cccccc',
            padding: '40px 24px'
          }}>
            <img 
              src={logo} 
              alt="IAbel Logo" 
              style={{
                width: '120px',
                height: '120px',
                borderRadius: '16px',
                marginBottom: '32px',
                objectFit: 'cover'
              }}
            />
            <h2 style={{ 
              margin: '0 0 16px 0', 
              color: '#ffffff',
              fontSize: '27px',
              fontWeight: 600
            }}>
              Como posso ajudá-lo hoje?
            </h2>
            <div style={{
              marginTop: '48px',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '12px',
              maxWidth: '800px',
              marginBottom: '40px'
            }}>
              {(() => {
                const suggestions = {
                  'rag_v1': [
                    'O que é INSIM?',
                    'Como funciona waterflooding?',
                    'Definição de INSIM-FT',
                    'O que são modelos capacitivos-resistivos?'
                  ],
                  'rag_v2': [
                    'Análise avançada do INSIM',
                    'Comparação entre métodos de simulação',
                    'Otimização de parâmetros de reservatório',
                    'Técnicas avançadas de waterflooding'
                  ],
                  'lora_only': [
                    'Explique sobre engenharia de reservatórios',
                    'Como funcionam simulações numéricas?',
                    'O que são métodos de recuperação?',
                    'Descreva modelagem de reservatórios'
                  ],
                  'hybrid': [
                    'Compare métodos de simulação',
                    'Análise detalhada do INSIM',
                    'Vantagens do waterflooding',
                    'Como escolher método de EOR?'
                  ]
                };
                return suggestions[chatMode] || suggestions['rag_v1'];
              })().map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => setInputValue(suggestion)}
                  style={{
                    padding: '16px 20px',
                    backgroundColor: '#2a2a2a',
                    border: '1px solid #555',
                    borderRadius: '12px',
                    cursor: 'pointer',
                    fontSize: '15px',
                    color: '#cccccc',
                    textAlign: 'left',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#3a3a3a';
                    e.currentTarget.style.borderColor = '#666';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#2a2a2a';
                    e.currentTarget.style.borderColor = '#555';
                    e.currentTarget.style.transform = 'translateY(0)';
                  }}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            style={{
              display: 'flex',
              justifyContent: message.isUser ? 'flex-end' : 'flex-start',
              marginBottom: '8px'
            }}
          >
            <div style={{
              maxWidth: '70%',
              display: 'flex',
              flexDirection: message.isUser ? 'row-reverse' : 'row',
              alignItems: 'flex-start',
              gap: '8px'
            }}>
              {!message.isUser && (
                <img 
                  src={thinkingLogo} 
                  alt="IAbel" 
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '6px',
                    objectFit: 'cover',
                    marginTop: '4px'
                  }}
                />
              )}
              
              <div style={{
                backgroundColor: message.isUser ? '#2c7dd3ff' : '#2a2a2a',
                color: message.isUser ? 'white' : '#ffffff',
                padding: '12px 16px',
                borderRadius: message.isUser ? '18px 18px 4px 18px' : '18px 18px 18px 4px',
                fontSize: '15px',
                lineHeight: '1.4',
                border: message.isUser ? 'none' : '1px solid #555',
                boxShadow: message.isUser ? '0 2px 8px rgba(0,123,255,0.15)' : '0 1px 3px rgba(0,0,0,0.3)',
                whiteSpace: 'pre-wrap'
              }}>
                {message.text}
                
                {!message.isUser && (message.confidence !== undefined || message.generationMode) && (
                  <div style={{
                    marginTop: '8px',
                    fontSize: '12px',
                    color: '#999',
                    borderTop: '1px solid #444',
                    paddingTop: '8px',
                    display: 'flex',
                    gap: '16px',
                    flexWrap: 'wrap'
                  }}>
                    {message.confidence !== undefined && (
                      <span>Confiança: {(message.confidence * 100).toFixed(1)}%</span>
                    )}
                    {message.sources !== undefined && (
                      <span>Fontes: {message.sources}</span>
                    )}
                    {message.generationMode && (
                      <span style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '4px',
                        padding: '2px 6px',
                        backgroundColor: '#333',
                        borderRadius: '4px',
                        fontSize: '11px'
                      }}>
                        {(() => {
                          if (message.generationMode.includes('rag')) return '📚';
                          if (message.generationMode.includes('lora')) return '🧠';
                          if (message.generationMode.includes('hybrid')) return '⚡';
                          if (message.generationMode.includes('adaptive')) return '🎯';
                          return '💬';
                        })()}
                        {message.generationMode.replace(/_/g, ' ').toUpperCase()}
                      </span>
                    )}
                  </div>
                )}
              </div>
              
              {message.isUser && (
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  backgroundColor: '#6b7280',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '14px',
                  fontWeight: 500,
                  marginTop: '4px'
                }}>
                  U
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div style={{
            display: 'flex',
            justifyContent: 'flex-start',
            marginBottom: '8px'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '8px'
            }}>
              <img 
                src={thinkingLogo} 
                alt="IAbel pensando" 
                style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '6px',
                  objectFit: 'cover',
                  marginTop: '4px'
                }}
              />
              <div style={{
                backgroundColor: '#2a2a2a',
                border: '1px solid #555',
                borderRadius: '18px 18px 18px 4px',
                padding: '12px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px'
              }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: '#94a3b8',
                  animation: 'pulse 1.5s ease-in-out infinite'
                }} />
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: '#94a3b8',
                  animation: 'pulse 1.5s ease-in-out infinite 0.2s'
                }} />
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: '#94a3b8',
                  animation: 'pulse 1.5s ease-in-out infinite 0.4s'
                }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        backgroundColor: messages.length === 0 ? 'transparent' : '#44484b',
        borderTop: messages.length === 0 ? 'none' : '1px solid #555',
        padding: messages.length === 0 ? '0 24px 40px 24px' : '16px 24px',
        display: 'flex',
        justifyContent: 'center'
      }}>
        <div style={{
          width: '100%',
          maxWidth: messages.length === 0 ? '700px' : '100%',
          position: 'relative'
        }}>
          <div style={{
            position: 'relative',
            display: 'flex',
            alignItems: 'flex-end',
            backgroundColor: '#2a2a2a',
            border: '1px solid #555',
            borderRadius: '24px',
            padding: '4px',
            transition: 'all 0.2s'
          }}>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Mensagem IAbel..."
              disabled={isLoading}
              style={{
                flex: 1,
                minHeight: '24px',
                maxHeight: '120px',
                padding: '12px 16px',
                border: 'none',
                borderRadius: '20px',
                fontSize: '15px',
                lineHeight: '1.4',
                resize: 'none',
                outline: 'none',
                fontFamily: 'inherit',
                backgroundColor: 'transparent',
                color: isLoading ? '#999' : '#ffffff',
                boxSizing: 'border-box'
              }}
              onFocus={(e) => {
                if (e.target.parentElement) {
                  e.target.parentElement.style.borderColor = '#007bff';
                  e.target.parentElement.style.boxShadow = '0 0 0 2px rgba(0,123,255,0.2)';
                }
              }}
              onBlur={(e) => {
                if (e.target.parentElement) {
                  e.target.parentElement.style.borderColor = '#555';
                  e.target.parentElement.style.boxShadow = 'none';
                }
              }}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '18px',
                border: 'none',
                backgroundColor: inputValue.trim() && !isLoading ? '#ffffff' : '#555',
                cursor: inputValue.trim() && !isLoading ? 'pointer' : 'not-allowed',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginLeft: '8px',
                transition: 'all 0.2s',
                flexShrink: 0
              }}
              onMouseEnter={(e) => {
                if (inputValue.trim() && !isLoading) {
                  e.currentTarget.style.backgroundColor = '#f0f0f0';
                }
              }}
              onMouseLeave={(e) => {
                if (inputValue.trim() && !isLoading) {
                  e.currentTarget.style.backgroundColor = '#ffffff';
                }
              }}
            >
              <img 
                src={upArrow} 
                alt="Enviar" 
                style={{
                  width: '16px',
                  height: '16px',
                  opacity: inputValue.trim() && !isLoading ? 1 : 0.5,
                  filter: inputValue.trim() && !isLoading ? 'invert(0)' : 'invert(1)'
                }}
              />
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default App;