from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import os

# ------------------------------
# Flask App Configuration
# ------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# --- MySQL Configuration (XAMPP default: username=root, no password) ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost:3306/chronic_disease_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# ------------------------------
# Database Model
# ------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Create tables if not exist
with app.app_context():
    db.create_all()

# ------------------------------
# Load Trained Model
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully.")
else:
    print("⚠️ Model file 'best_model.pkl' not found.")
    model = None

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('predict_page'))
    return render_template('home.html')


# ------------------------------
# Registration Route
# ------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('register.html', error='Username already exists.')

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('register.html')


# ------------------------------
# Login Route
# ------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']

        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('predict_page'))
        else:
            return render_template('login.html', error='Invalid username or password.')
    return render_template('login.html')


# ------------------------------
# Forgot Password Route
# ------------------------------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        message = f"A password reset link has been sent to {email} (demo only)."
        return render_template('forgot_password.html', message=message)
    return render_template('forgot_password.html')


# ------------------------------
# Logout Route
# ------------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))


# ------------------------------
# Prediction Route
# ------------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if not model:
            return render_template('predict.html', prediction_text="Model not loaded.", report_data={})

        try:
            form_data = {
                'age': [float(request.form['age'])],
                'gender': [int(request.form['gender'])],
                'bmi': [float(request.form['bmi'])],
                'blood_pressure': [float(request.form['blood_pressure'])],
                'cholesterol_level': [float(request.form['cholesterol_level'])],
                'glucose_level': [float(request.form['glucose_level'])],
                'physical_activity': [float(request.form['physical_activity'])],
                'smoking_status': [int(request.form['smoking_status'])],
                'alcohol_intake': [float(request.form['alcohol_intake'])],
                'family_history': [int(request.form['family_history'])],
                'biomarker_A': [float(request.form.get('biomarker_A', 0))],
                'biomarker_B': [float(request.form.get('biomarker_B', 0))],
                'biomarker_C': [float(request.form.get('biomarker_C', 0))],
                'biomarker_D': [float(request.form.get('biomarker_D', 0))]
            }

            input_df = pd.DataFrame(form_data)
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)
            confidence = float(prediction_proba.max()) * 100

            disease_map = {
                0: 'No Disease',
                1: 'Heart Disease',
                2: 'Diabetes',
                3: 'Cancer',
                4: 'Hypertension'
            }

            result_text = f"Prediction: {disease_map.get(prediction, 'Unknown')} with {confidence:.2f}% confidence."

            return render_template('predict.html', prediction_text=result_text, report_data=form_data)

        except Exception as e:
            return render_template('predict.html', prediction_text=f"Error: {e}", report_data={})

    return render_template('predict.html', report_data={})


# ------------------------------
# Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
