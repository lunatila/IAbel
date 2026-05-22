"""
Logging configuration for IAbel Enhanced RAG API
Provides structured logging with different levels and output formats
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Dict, Any, List


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with JSON format (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    
    # Create application-specific logger
    app_logger = logging.getLogger('iabel')
    
    return app_logger


def log_with_context(logger: logging.Logger, level: str, message: str, **context):
    """
    Log message with additional context data
    
    Args:
        logger: Logger instance
        level: Log level (info, warning, error, etc.)
        message: Log message
        **context: Additional context data
    """
    # Create a log record with extra data
    log_method = getattr(logger, level.lower())
    
    # Create a custom log record
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        '',
        0,
        message,
        (),
        None
    )
    
    # Add context as extra data
    record.extra_data = context
    
    # Log the record
    logger.handle(record)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"iabel.{self.__class__.__name__}")
        return self._logger
    
    def log_operation(self, operation: str, **context):
        """Log the start of an operation"""
        self.logger.info(f"Starting {operation}", extra={'extra_data': context})
    
    def log_success(self, operation: str, duration: float = None, **context):
        """Log successful operation completion"""
        if duration:
            context['duration_ms'] = round(duration * 1000, 2)
        self.logger.info(f"Completed {operation}", extra={'extra_data': context})
    
    def log_error(self, operation: str, error: Exception, **context):
        """Log operation error"""
        context.update({
            'error_type': type(error).__name__,
            'error_message': str(error)
        })
        self.logger.error(f"Failed {operation}", extra={'extra_data': context}, exc_info=True)
    
    def log_warning(self, message: str, **context):
        """Log warning with context"""
        self.logger.warning(message, extra={'extra_data': context})


# Performance logging decorator
def log_performance(operation_name: str = None):
    """
    Decorator to log function performance
    
    Args:
        operation_name: Custom operation name (defaults to function name)
    """
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = logging.getLogger(f"iabel.performance")
            
            start_time = time.time()
            try:
                logger.info(f"Starting {op_name}")
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {op_name}", extra={
                    'extra_data': {
                        'duration_ms': round(duration * 1000, 2),
                        'success': True
                    }
                })
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {op_name}", extra={
                    'extra_data': {
                        'duration_ms': round(duration * 1000, 2),
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'success': False
                    }
                }, exc_info=True)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = logging.getLogger(f"iabel.performance")
            
            start_time = time.time()
            try:
                logger.info(f"Starting {op_name}")
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {op_name}", extra={
                    'extra_data': {
                        'duration_ms': round(duration * 1000, 2),
                        'success': True
                    }
                })
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {op_name}", extra={
                    'extra_data': {
                        'duration_ms': round(duration * 1000, 2),
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'success': False
                    }
                }, exc_info=True)
                raise
        
        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Error tracking utilities
class ErrorTracker:
    """
    Track and analyze errors in the application
    """
    
    def __init__(self):
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """
        Track an error occurrence
        
        Args:
            error: Exception instance
            context: Additional context about the error
        """
        error_type = type(error).__name__
        error_key = f"{error_type}: {str(error)[:100]}"
        
        # Update error counts
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to recent errors
        error_info = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': error_type,
            'message': str(error),
            'context': context or {}
        }
        
        self.recent_errors.append(error_info)
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked errors
        """
        return {
            'total_unique_errors': len(self.error_counts),
            'total_error_occurrences': sum(self.error_counts.values()),
            'most_common_errors': sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'recent_errors_count': len(self.recent_errors)
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent errors
        
        Args:
            limit: Maximum number of recent errors to return
        """
        return self.recent_errors[-limit:]


# Global error tracker instance
error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance"""
    return error_tracker