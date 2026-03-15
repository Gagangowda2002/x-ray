"""
Configuration file for Flask application
"""
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
    HEATMAP_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'heatmaps')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'dicom', 'dcm'}
    
    # Model
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'densenet_model.h5')
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'app.log')
    
    # Session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    SESSION_COOKIE_SECURE = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    WTF_CSRF_ENABLED = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
