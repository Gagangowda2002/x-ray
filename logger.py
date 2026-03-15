"""
Logging configuration module
"""
import logging
import logging.handlers
import os
from config import Config

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create application logger
app_logger = setup_logger(
    'xray_app',
    log_file=Config.LOG_FILE,
    level=getattr(logging, Config.LOG_LEVEL)
)

def get_logger(name):
    """Get logger by name"""
    return logging.getLogger(name)
