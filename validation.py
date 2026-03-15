"""
Validation utilities for file uploads and predictions
"""
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from logger import get_logger

logger = get_logger(__name__)

def allowed_file(filename, allowed_extensions):
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions
    
    Returns:
        bool: True if file is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_upload_file(file, allowed_extensions, max_size):
    """
    Validate uploaded file
    
    Args:
        file: FileStorage object from request
        allowed_extensions: Set of allowed extensions
        max_size: Maximum file size in bytes
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check if file is present
        if not file or file.filename == '':
            return False, "No file selected"
        
        # Check file extension
        if not allowed_file(file.filename, allowed_extensions):
            return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
        
        # Get secure filename
        filename = secure_filename(file.filename)
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size == 0:
            return False, "File is empty"
        
        if file_size > max_size:
            max_size_mb = max_size / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_size_mb:.1f}MB"
        
        logger.info(f"File validation successful: {filename} ({file_size} bytes)")
        return True, filename
        
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_image_shape(image_array, expected_shape):
    """
    Validate image shape before prediction
    
    Args:
        image_array: Numpy array of image
        expected_shape: Expected shape
    
    Returns:
        bool: True if shape matches, False otherwise
    """
    try:
        if image_array.shape != expected_shape:
            logger.warning(f"Image shape mismatch. Expected: {expected_shape}, Got: {image_array.shape}")
            return False
        return True
    except Exception as e:
        logger.error(f"Image shape validation error: {str(e)}")
        return False
