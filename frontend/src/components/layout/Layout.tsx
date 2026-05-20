import React from 'react';
import { motion } from 'framer-motion';
import { 
  MessageCircle, 
  Search, 
  Upload, 
  BarChart3, 
  Moon, 
  Sun,
  Menu,
  X
} from 'lucide-react';
import { useAppStore } from '../../stores/appStore';
import Button from '../ui/Button';
import Badge from '../ui/Badge';
import logoTransparent from '../../images/IAbel_transp.png';

interface LayoutProps {
  children: React.ReactNode;
  currentPage: string;
  onPageChange: (page: string) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, currentPage, onPageChange }) => {
  const { theme, toggleTheme, systemStatus } = useAppStore();
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  const navigation = [
    {
      name: 'Chat',
      id: 'chat',
      icon: MessageCircle,
      description: 'Converse com IAbel'
    },
    {
      name: 'Buscar',
      id: 'search',
      icon: Search,
      description: 'Buscar documentos'
    },
    {
      name: 'Upload',
      id: 'upload',
      icon: Upload,
      description: 'Carregar PDFs'
    },
    {
      name: 'Status',
      id: 'status',
      icon: BarChart3,
      description: 'Status do sistema'
    }
  ];

  const currentNav = navigation.find(nav => nav.id === currentPage);

  React.useEffect(() => {
    // Apply theme to document
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [theme]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white dark:bg-gray-800 shadow-lg transform transition-transform duration-200 ease-in-out
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 flex items-center justify-center">
                <img 
                  src={logoTransparent} 
                  alt="IAbel Logo" 
                  className="w-8 h-8 object-contain"
                />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">
                  IAbel
                </h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  v2.0.0
                </p>
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              className="lg:hidden"
              onClick={() => setSidebarOpen(false)}
              icon={<X size={16} />}
            />
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;

              return (
                <motion.button
                  key={item.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => {
                    onPageChange(item.id);
                    setSidebarOpen(false);
                  }}
                  className={`
                    w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-all duration-200
                    ${isActive 
                      ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800' 
                      : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }
                  `}
                >
                  <Icon size={20} />
                  <div className="flex-1 min-w-0">
                    <div className="font-medium">{item.name}</div>
                    <div className="text-xs opacity-75">{item.description}</div>
                  </div>
                </motion.button>
              );
            })}
          </nav>

          {/* System Status */}
          {systemStatus && (
            <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    Sistema
                  </span>
                  <Badge 
                    variant={systemStatus.llm.status === 'online' ? 'success' : 'error'} 
                    size="sm"
                  >
                    {systemStatus.llm.status === 'online' ? 'Online' : 'Offline'}
                  </Badge>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    Documentos
                  </span>
                  <span className="text-xs font-mono text-gray-700 dark:text-gray-300">
                    {systemStatus.vector_store.total_documents}
                  </span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    Cache
                  </span>
                  <Badge 
                    variant={systemStatus.cache.connected ? 'success' : 'warning'} 
                    size="sm"
                  >
                    {systemStatus.cache.cache_type}
                  </Badge>
                </div>
              </div>
            </div>
          )}

          {/* Footer */}
          <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Tema
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleTheme}
                icon={theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
              >
                {theme === 'dark' ? 'Escuro' : 'Claro'}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="w-full">
        {/* Top bar */}
        <div className="sticky top-0 z-30 bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarOpen(true)}
                icon={<Menu size={16} />}
              />
              
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  {currentNav?.name}
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {currentNav?.description}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* Theme toggle for desktop */}
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleTheme}
                icon={theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
                className="hidden sm:flex"
              />

              {/* Status indicators */}
              {systemStatus && (
                <div className="hidden md:flex items-center space-x-2">
                  <Badge 
                    variant={systemStatus.llm.status === 'online' ? 'success' : 'error'} 
                    size="sm"
                  >
                    {systemStatus.llm.status === 'online' ? 'LLM Online' : 'LLM Offline'}
                  </Badge>
                  
                  <Badge 
                    variant={systemStatus.cache.connected ? 'success' : 'warning'} 
                    size="sm"
                  >
                    Cache {systemStatus.cache.connected ? 'OK' : 'Warn'}
                  </Badge>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className={currentPage === 'chat' ? 'p-2 lg:p-4' : 'p-4 lg:p-6'}>
          <motion.div
            key={currentPage}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="max-w-7xl mx-auto"
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  );
};

export default Layout;