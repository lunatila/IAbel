import React from 'react';
import { motion } from 'framer-motion';
import logo from '../../images/IAbel_transp.png';
import Button from '../ui/Button';

interface WelcomeScreenProps {
  onSuggestedQuestion: (question: string) => void;
  ragMode: 'rag_v1' | 'rag_v2' | 'rag_v3';
  onRagModeChange: (mode: 'rag_v1' | 'rag_v2' | 'rag_v3') => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  onSuggestedQuestion,
  ragMode,
  onRagModeChange
}) => {
  return (
    <div className="flex flex-col items-center gap-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="flex items-center gap-3"
      >
        <img src={logo} alt="Logo" className="w-8 h-8 object-contain" />
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">
          Como posso ajudar você hoje?
        </h1>
      </motion.div>

      {/* RAG Version Selector */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="flex flex-col items-center gap-3"
      >
        <span className="text-sm text-gray-600 dark:text-gray-400">
          Escolha a versão do RAG:
        </span>
        <div className="flex gap-2">
          <Button
            variant={ragMode === 'rag_v1' ? 'primary' : 'outline'}
            size="sm"
            onClick={() => onRagModeChange('rag_v1')}
            className="px-4 py-2"
          >
            <div className="text-center">
              <div className="font-semibold">v1</div>
              <div className="text-xs opacity-75">Básico</div>
            </div>
          </Button>
          <Button
            variant={ragMode === 'rag_v2' ? 'primary' : 'outline'}
            size="sm"
            onClick={() => onRagModeChange('rag_v2')}
            className="px-4 py-2"
          >
            <div className="text-center">
              <div className="font-semibold">v2</div>
              <div className="text-xs opacity-75">Avançado</div>
            </div>
          </Button>
          <Button
            variant={ragMode === 'rag_v3' ? 'primary' : 'outline'}
            size="sm"
            onClick={() => onRagModeChange('rag_v3')}
            className="px-4 py-2"
          >
            <div className="text-center">
              <div className="font-semibold">v3</div>
              <div className="text-xs opacity-75">English + Citations</div>
            </div>
          </Button>
        </div>
      </motion.div>
    </div>
  );
};

export default WelcomeScreen;
