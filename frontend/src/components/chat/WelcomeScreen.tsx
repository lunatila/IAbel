import React from 'react';
import { motion } from 'framer-motion';
import logo from '../../images/IAbel_transp.png';

interface WelcomeScreenProps {
  onSuggestedQuestion: (question: string) => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = () => {
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
    </div>
  );
};

export default WelcomeScreen;
