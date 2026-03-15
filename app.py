
"""
Main Flask application factory
"""
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from logger import setup_logger, get_logger
from config import config
from prediction import PredictionEngine
from validation import validate_upload_file, validate_image_shape
from model.preprocessing import preprocess_image
from model.gradcam import generate_gradcam, overlay_heatmap

logger = get_logger(__name__)

def create_app(config_name='development'):
    """
    Application factory function
    
    Args:
        config_name: Configuration environment name
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    # Default production configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'
app.config['MODEL_PATH'] = 'model/densenet_model.h5'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['LOG_FILE'] = 'logs/app.log'
app.config['LOG_LEVEL'] = 'INFO'
    
    # Setup logger
    setup_logger(
        'xray_app',
        log_file=app.config.get('LOG_FILE'),
        level=app.config.get('LOG_LEVEL', 'INFO')
    )
    
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config.get('LOG_FILE', '')), exist_ok=True)
    
    # Initialize prediction engine
    try:
        prediction_engine = PredictionEngine(app.config['MODEL_PATH'])
        logger.info("Prediction engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize prediction engine: {str(e)}")
        prediction_engine = None
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        logger.warning(f"404 error: {request.path}")
        return render_template('error.html', error_code=404, message='Page not found'), 404
    
    @app.errorhandler(500)
    def server_error(error):
        """Handle 500 errors"""
        logger.error(f"500 error: {str(error)}")
        return render_template('error.html', error_code=500, message='Internal server error'), 500
    
    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_file(error):
        """Handle file too large errors"""
        max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        message = f"File too large. Maximum size: {max_size_mb:.1f}MB"
        logger.warning(f"File too large error: {message}")
        return render_template('error.html', error_code=413, message=message), 413
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle all other exceptions"""
        logger.error(f"Unhandled exception: {str(error)}")
        return render_template('error.html', error_code=500, message='An unexpected error occurred'), 500
    
    # Home route
    @app.route('/', methods=['GET', 'POST'])
    def index():
        """Main page with file upload"""
        try:
            if request.method == 'POST':
                logger.info("Processing image upload request")
                
                # Validate file
                if 'file' not in request.files:
                    logger.warning("No file part in request")
                    return render_template('index.html', error='No file selected'), 400
                
                file = request.files['file']
                is_valid, filename_or_error = validate_upload_file(
                    file,
                    app.config['ALLOWED_EXTENSIONS'],
                    app.config['MAX_CONTENT_LENGTH']
                )
                
                if not is_valid:
                    logger.warning(f"File validation failed: {filename_or_error}")
                    return render_template('index.html', error=filename_or_error), 400
                
                filename = filename_or_error
                
                # Save uploaded file
                try:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    logger.info(f"File uploaded and saved: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save file: {str(e)}")
                    return render_template('index.html', error=f"Failed to save file: {str(e)}"), 500
                
                # Preprocess image
                try:
                    logger.info(f"Preprocessing image: {filepath}")
                    img = preprocess_image(filepath)
                    logger.info(f"Image preprocessed successfully: {filepath}")
                except Exception as e:
                    logger.error(f"Image preprocessing failed: {str(e)}", exc_info=True)
                    return render_template('index.html', error=f"Failed to process image: {str(e)}"), 400
                
                # Make prediction
                if prediction_engine is None:
                    logger.error("Prediction engine not initialized")
                    return render_template('index.html', error='Model not available'), 500
                
                try:
                    logger.info("Making prediction...")
                    prediction_result = prediction_engine.predict(img)
                    label = prediction_result['class']
                    confidence = prediction_result['confidence']
                    logger.info(f"Prediction made: {label} ({confidence:.2f}%)")
                except Exception as e:
                    logger.error(f"Prediction failed: {str(e)}", exc_info=True)
                    return render_template('index.html', error=f"Prediction failed: {str(e)}"), 500
                
                # Generate Grad-CAM heatmap (non-critical, can fail gracefully)
                heatmap_relative = None
                try:
                    logger.info("Generating Grad-CAM heatmap...")
                    heatmap = generate_gradcam(prediction_engine.model, img)
                    heatmap_filename = f"heatmap_{os.path.splitext(filename)[0]}.png"
                    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], heatmap_filename)
                    overlay_heatmap(filepath, heatmap, heatmap_path)
                    logger.info(f"Heatmap generated: {heatmap_path}")
                    heatmap_relative = os.path.join('heatmaps', heatmap_filename)
                except Exception as e:
                    logger.warning(f"Heatmap generation failed (non-critical): {str(e)}")
                    heatmap_relative = None
                
                logger.info(f"Image processing completed successfully for: {filename}")
                
                return render_template(
                    'result.html',
                    label=label,
                    confidence=round(confidence, 2),
                    image=os.path.join('uploads', filename),
                    heatmap=heatmap_relative,
                    top_3=prediction_result.get('top_3_predictions', [])
                )
            
            return render_template('index.html')
        
        except Exception as e:
            logger.error(f"Unhandled error in index route: {str(e)}", exc_info=True)
            return render_template('index.html', error='An error occurred during processing. Please try again.'), 500
    
    # API endpoint for predictions
    @app.route('/api/predict', methods=['POST'])
    def api_predict():
        """REST API endpoint for predictions"""
        try:
            logger.info("API prediction request received")
            
            if 'file' not in request.files:
                logger.warning("API: No file part in request")
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            is_valid, filename_or_error = validate_upload_file(
                file,
                app.config['ALLOWED_EXTENSIONS'],
                app.config['MAX_CONTENT_LENGTH']
            )
            
            if not is_valid:
                logger.warning(f"API: File validation failed: {filename_or_error}")
                return jsonify({'error': filename_or_error}), 400
            
            filename = filename_or_error
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                logger.info(f"API: File saved: {filepath}")
            except Exception as e:
                logger.error(f"API: Failed to save file: {str(e)}")
                return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
            
            # Preprocess and predict
            try:
                img = preprocess_image(filepath)
            except Exception as e:
                logger.error(f"API: Image preprocessing failed: {str(e)}", exc_info=True)
                return jsonify({'error': f"Failed to process image: {str(e)}"}), 400
            
            if prediction_engine is None:
                logger.error("API: Prediction engine not initialized")
                return jsonify({'error': 'Model not available'}), 500
            
            try:
                prediction_result = prediction_engine.predict(img)
                logger.info(f"API: Prediction made: {prediction_result['class']}")
            except Exception as e:
                logger.error(f"API: Prediction failed: {str(e)}", exc_info=True)
                return jsonify({'error': f"Prediction failed: {str(e)}"}), 500
            
            return jsonify({
                'success': True,
                'class': prediction_result['class'],
                'confidence': prediction_result['confidence'],
                'confidence_score': prediction_result['confidence_score'],
                'top_3_predictions': prediction_result['top_3_predictions'],
                'probabilities': prediction_result['probabilities']
            }), 200
        
        except Exception as e:
            logger.error(f"API: Unhandled error in predict: {str(e)}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred'}), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        status = {
            'status': 'healthy',
            'model_loaded': prediction_engine is not None
        }
        return jsonify(status), 200
    
    # Info endpoint
    @app.route('/api/info', methods=['GET'])
    def api_info():
        """Get API information"""
        return jsonify({
            'app_name': 'X-Ray Medical Image Classification',
            'version': '1.0.0',
            'model': 'DenseNet',
            'classes': prediction_engine.get_class_names() if prediction_engine else [],
            'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024),
            'allowed_extensions': list(app.config['ALLOWED_EXTENSIONS'])
        }), 200
    
    logger.info("Flask application created successfully")
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
