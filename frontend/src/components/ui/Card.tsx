import React from 'react';
import { clsx } from 'clsx';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  variant?: 'default' | 'bordered' | 'elevated';
}

const Card: React.FC<CardProps> = ({
  children,
  className,
  padding = 'md',
  variant = 'default'
}) => {
  const baseStyles = 'bg-white dark:bg-gray-800 rounded-lg';
  
  const variants = {
    default: '',
    bordered: 'border border-gray-200 dark:border-gray-700',
    elevated: 'shadow-lg'
  };

  const paddings = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6'
  };

  return (
    <div
      className={clsx(
        baseStyles,
        variants[variant],
        paddings[padding],
        className
      )}
    >
      {children}
    </div>
  );
};

export default Card;