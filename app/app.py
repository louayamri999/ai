import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'skinai_secret_encryption_key_for_sessions'

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Database connection helper
DB_PATH = os.path.join(app.root_path, 'skinai.db')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Database initialization
def init_db():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        
        # Create patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Seed default admin user if not exists
        cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
        if cursor.fetchone() is None:
            # Seed default admin
            pw_hash = generate_password_hash('admin123')
            cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', ('admin', pw_hash))
            conn.commit()

# Initialize DB on startup
init_db()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# TensorFlow Model lazy-loading
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(app.root_path, '..', 'model', 'vgg16_malignant_vs_benign.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load model using TensorFlow
        # Importing tensorflow inside the function keeps startup fast for routes that don't need it
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    
    # Get statistics
    total_count = db.execute('SELECT COUNT(*) FROM patients').fetchone()[0]
    malignant_count = db.execute("SELECT COUNT(*) FROM patients WHERE prediction = 'Malignant'").fetchone()[0]
    benign_count = db.execute("SELECT COUNT(*) FROM patients WHERE prediction = 'Benign'").fetchone()[0]
    
    # Calculate risk percentage
    risk_percentage = 0
    if total_count > 0:
        risk_percentage = round((malignant_count / total_count) * 100, 1)
        
    # Get recent patients
    recent_patients = db.execute('SELECT * FROM patients ORDER BY created_at DESC LIMIT 5').fetchall()
    
    return render_template('dashboard.html', 
                           total_count=total_count,
                           malignant_count=malignant_count,
                           benign_count=benign_count,
                           risk_percentage=risk_percentage,
                           recent_patients=recent_patients)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Get patient info
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        
        # Verify file upload
        if 'image' not in request.files:
            flash('No file selected.', 'error')
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            # Create a unique filename using timestamp
            import time
            unique_filename = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            try:
                # Preprocess image using the exact code from your Google Colab
                import tensorflow as tf
                img = tf.keras.preprocessing.image.load_img(
                          filepath, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Load model and predict
                model_inst = get_model()
                pred_raw = model_inst.predict(img_array)
                pred_val = float(pred_raw[0][0])
                
                # Decision boundary from Precision-Recall curve
                threshold = 0.2976
                
                if pred_val > threshold:
                    prediction = 'Malignant'
                    risk_level = 'High Risk'
                    confidence = pred_val * 100
                else:
                    prediction = 'Benign'
                    risk_level = 'Low Risk'
                    confidence = (1.0 - pred_val) * 100
                
                # Save to database
                db = get_db()
                db.execute('''
                    INSERT INTO patients (name, age, gender, image_path, prediction, confidence, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (name, age, gender, unique_filename, prediction, round(confidence, 2), risk_level))
                db.commit()
                
                # Get the patient record ID
                patient_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
                
                return redirect(url_for('result', patient_id=patient_id))
                
            except Exception as e:
                # Remove file if something failed
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid image format. Allowed formats: PNG, JPG, JPEG, WEBP.', 'error')
            
    return render_template('predict.html')

@app.route('/result/<int:patient_id>')
@login_required
def result(patient_id):
    db = get_db()
    patient = db.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    
    if patient is None:
        flash('Patient record not found.', 'error')
        return redirect(url_for('dashboard'))
        
    return render_template('result.html', patient=patient)

@app.route('/patients')
@login_required
def patients():
    search_query = request.args.get('search', '').strip()
    risk_filter = request.args.get('risk', '').strip()
    
    db = get_db()
    
    query = 'SELECT * FROM patients WHERE 1=1'
    params = []
    
    if search_query:
        query += ' AND name LIKE ?'
        params.append(f'%{search_query}%')
        
    if risk_filter:
        query += ' AND risk_level = ?'
        params.append(risk_filter)
        
    query += ' ORDER BY created_at DESC'
    
    patient_records = db.execute(query, params).fetchall()
    
    return render_template('patients.html', 
                           patients=patient_records,
                           search=search_query,
                           risk=risk_filter)

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    db = get_db()
    patient = db.execute('SELECT * FROM patients WHERE id = ?', (patient_id,)).fetchone()
    
    if patient:
        # Delete image file from disk
        image_name = patient['image_path']
        image_file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        if os.path.exists(image_file_path):
            try:
                os.remove(image_file_path)
            except Exception as e:
                print(f"Error deleting file {image_file_path}: {e}")
                
        # Delete record from database
        db.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
        db.commit()
        flash('Patient record deleted successfully.', 'success')
    else:
        flash('Patient record not found.', 'error')
        
    return redirect(url_for('patients'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
