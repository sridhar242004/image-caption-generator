from flask import Flask, flash, request, redirect, url_for, render_template, session
import os
import numpy as np
from PIL import Image
import pickle
from datetime import datetime
import traceback
import logging
import tensorflow as tf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Constants
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create required directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize variables
model = None
tokenizer = None
xception_model = None

def load_models():
    global model, tokenizer, xception_model
    
    logger.info("Loading TensorFlow...")
    try:
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        # Suppress TensorFlow logging
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        logger.info("Loading tokenizer...")
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.p')
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer file not found at {tokenizer_path}")
            return False
            
        with open(tokenizer_path, 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        
        logger.info("Loading caption model...")
        model_path = os.path.join(os.path.dirname(__file__), 'model_9.h5')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
            
        model = tf.keras.models.load_model(model_path, compile=False)
        
        logger.info("Loading Xception model...")
        xception_model = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        
        logger.info("All models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")
        
        # Load and preprocess image
        img = Image.open(image_path)
        logger.info(f"Image format: {img.format}, size: {img.size}, mode: {img.mode}")
        
        img = img.convert('RGB')
        img = img.resize((299, 299))
        img_array = np.array(img)
        
        # Expand dimensions
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 127.5 - 1.0
        
        logger.info("Image processing completed successfully")
        return img_array
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_caption(image_array):
    try:
        logger.info("Generating caption...")
        
        # Extract features
        features = xception_model.predict(image_array, verbose=0)
        logger.info("Features extracted successfully")
        
        # Generate caption
        max_length = 32
        in_text = 'start'
        
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence], 
                maxlen=max_length,
                padding='post'
            )
            
            pred = model.predict([features, sequence], verbose=0)
            pred = np.argmax(pred)
            
            word = None
            for w, idx in tokenizer.word_index.items():
                if idx == pred:
                    word = w
                    break
                    
            if word is None or word == 'end':
                break
                
            in_text += ' ' + word
        
        caption = ' '.join(word for word in in_text.split() 
                         if word not in ['start', 'end'])
        caption = caption.capitalize()
        
        logger.info(f"Generated caption: {caption}")
        return caption
        
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if not model or not tokenizer or not xception_model:
            flash('Models not loaded properly. Please restart the application.', 'error')
            return redirect(url_for('home'))
            
        if 'image' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('home'))
            
        file = request.files['image']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))
            
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a JPG, PNG, or GIF file.', 'error')
            return redirect(url_for('home'))
        
        # Save file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"image_{timestamp}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(app.root_path, UPLOAD_FOLDER, filename)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Process image and generate caption
        image_array = process_image(filepath)
        caption = generate_caption(image_array)
        
        # Store results in session
        session['prediction'] = caption
        session['img_path'] = os.path.join('uploads', filename)
        
        return redirect(url_for('show_result'))
                             
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        logger.error(traceback.format_exc())
        flash('An error occurred while processing the image. Please check the application logs.', 'error')
        return redirect(url_for('home'))

@app.route('/result')
def show_result():
    return render_template('index.html',
                           prediction=session.get('prediction'),
                           img_path=session.get('img_path'))

if __name__ == '__main__':
    if load_models():
        logger.info("Starting Flask application...")
        app.run(debug=True)
    else:
        logger.error("Failed to load models. Please check the error messages above.")