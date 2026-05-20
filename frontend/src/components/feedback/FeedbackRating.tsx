import React, { useState } from 'react';
import { Star, ThumbsUp, ThumbsDown } from 'lucide-react';
import Button from '../ui/Button';

interface FeedbackRatingProps {
  onRatingSubmit: (rating: number, feedbackType: 'thumbs_up' | 'thumbs_down' | 'rating') => Promise<void>;
  disabled?: boolean;
  compact?: boolean;
}

export const FeedbackRating: React.FC<FeedbackRatingProps> = ({
  onRatingSubmit,
  disabled = false,
  compact = false
}) => {
  const [rating, setRating] = useState<number>(0);
  const [hoveredRating, setHoveredRating] = useState<number>(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleStarClick = async (selectedRating: number) => {
    if (disabled || isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      await onRatingSubmit(selectedRating, 'rating');
      setRating(selectedRating);
      setSubmitted(true);
    } catch (error) {
      console.error('Failed to submit rating:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleQuickFeedback = async (type: 'thumbs_up' | 'thumbs_down') => {
    if (disabled || isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      const quickRating = type === 'thumbs_up' ? 5 : 1;
      await onRatingSubmit(quickRating, type);
      setRating(quickRating);
      setSubmitted(true);
    } catch (error) {
      console.error('Failed to submit feedback:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
        <ThumbsUp className="h-4 w-4" />
        <span>Obrigado pelo feedback!</span>
      </div>
    );
  }

  if (compact) {
    return (
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => handleQuickFeedback('thumbs_up')}
          disabled={disabled || isSubmitting}
          className="p-1 h-6 w-6 text-gray-500 hover:text-green-600 dark:hover:text-green-400"
        >
          <ThumbsUp className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => handleQuickFeedback('thumbs_down')}
          disabled={disabled || isSubmitting}
          className="p-1 h-6 w-6 text-gray-500 hover:text-red-600 dark:hover:text-red-400"
        >
          <ThumbsDown className="h-3 w-3" />
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-4">
        {/* Star Rating */}
        <div className="flex items-center gap-1">
          <span className="text-sm text-gray-600 dark:text-gray-400 mr-2">
            Avalie esta resposta:
          </span>
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onClick={() => handleStarClick(star)}
              onMouseEnter={() => setHoveredRating(star)}
              onMouseLeave={() => setHoveredRating(0)}
              disabled={disabled || isSubmitting}
              className="p-1 transition-colors disabled:opacity-50"
            >
              <Star
                className={`h-4 w-4 ${
                  star <= (hoveredRating || rating)
                    ? 'fill-yellow-400 text-yellow-400'
                    : 'text-gray-300 dark:text-gray-600'
                } transition-colors`}
              />
            </button>
          ))}
        </div>

        {/* Quick Feedback Buttons */}
        <div className="flex items-center gap-2 border-l pl-4 border-gray-200 dark:border-gray-700">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleQuickFeedback('thumbs_up')}
            disabled={disabled || isSubmitting}
            className="flex items-center gap-1 text-gray-600 hover:text-green-600 dark:text-gray-400 dark:hover:text-green-400"
          >
            <ThumbsUp className="h-4 w-4" />
            <span className="text-xs">Útil</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleQuickFeedback('thumbs_down')}
            disabled={disabled || isSubmitting}
            className="flex items-center gap-1 text-gray-600 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
          >
            <ThumbsDown className="h-4 w-4" />
            <span className="text-xs">Não útil</span>
          </Button>
        </div>
      </div>

      {isSubmitting && (
        <div className="text-xs text-gray-500 dark:text-gray-400">
          Enviando feedback...
        </div>
      )}
    </div>
  );
};