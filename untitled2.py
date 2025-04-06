import os
import re
import cv2
import numpy as np
import joblib
import pytesseract
import requests
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from script import analyze_blood_test, analyze_urine_test, analyze_thyroid_test, classify_cancer_type
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
from authlib.integrations.flask_client import OAuth
from geopy.geocoders import Nominatim
import google.generativeai as genai  # Gemini chatbot
from transformers import ViTImageProcessor, TFAutoModelForImageClassification
from pdf2image import convert_from_path
from PIL import Image
from werkzeug.security import check_password_hash
import secrets 
import joblib
import easyocr


# ----------------- Flask App Initialization---------------------
app = Flask(__name__, template_folder=os.path.abspath("templates") , static_folder="static")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key_here"

# ------------------------- Ensure Required Folders Exist------------------------
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------- Flask Extensions-------------------------
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # ✅ Add this line
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

#-------------------------Load Ml Models ------------------------------
 # Load AI Models
try:
    print("🔄 Loading AI models...")
    blood_model = joblib.load("models/blood_model.pkl")
    urine_model = joblib.load("models/urine_model.pkl")
    thyroid_model = joblib.load("models/thyroid_model.pkl")
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    
    
# OCR Reader
ocr_reader = easyocr.Reader(['en'])

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------- User Model----------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=True)  # Change nullable=False to nullable=True


    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)


# ------------------------------- Load User Function---------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Extract text from medical reports
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text

# Extract key medical values
def extract_medical_values(text):
    extracted_values = {}

    patterns = {
        "Hemoglobin": r'Hemoglobin:\s*([\d.]+)',
        "TSH": r'TSH:\s*([\d.]+)',
        "WBC": r'WBC:\s*([\d,]+)',
        "RBC": r'RBC:\s*([\d.]+)',
        "Platelet Count": r'Platelet Count:\s*([\d,]+)',
        "Creatinine": r'Creatinine:\s*([\d.]+)'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).replace(",", "")
            extracted_values[key] = float(value)

    return extracted_values


# API Endpoint to process reports
@app.route("/upload_report", methods=["POST"])
def process_report():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        extracted_text = extract_text_from_image(file_path)
        medical_values = extract_medical_values(extracted_text)     

        if not medical_values:
            return jsonify({"error": "No valid medical data found"}), 400

        # Determine which model to use
        if len(medical_values) >= 12:
            diagnosis_results = analyze_medical_data(medical_values, blood_model)
        elif len(medical_values) >= 10:
            diagnosis_results = analyze_medical_data(medical_values, urine_model)
        elif len(medical_values) >= 8:
            diagnosis_results = analyze_medical_data(medical_values, thyroid_model)
        else:
            return jsonify({"error": "Invalid report format"}), 400

        return jsonify({
            "extracted_data": medical_values,
            "diagnosis_results": diagnosis_results,
            "message": "Report processed successfully"
        })

    return jsonify({"error": "Invalid file type"}), 400

# ✅ Routes
@app.route("/")
def index():
    return render_template("index3.html")

@app.route("/hospital")
def hospital():
    return render_template("hospital.html")

@app.route("/analyse")
def analyse():
    return render_template("analyse.html")

@app.route("/chatbot2")
def chatbot_page():
    return render_template("chatbot2.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/Diet')
def Diet():
    return render_template('Diet.html')

@app.route('/redirect')
def redirect():
    return render_template('redirect.html')


@app.route("/dashboard-data")
@login_required
def dashboard_data():
    return jsonify({"username": current_user.username, "email": current_user.email})


@app.route("/login", methods=["GET", "POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):  # ✅ Correctly check hashed password
        login_user(user)
        return jsonify({"message": "Login successful", "username": user.username, "email": user.email}), 200

    return jsonify({"error": "Invalid email or password"}), 401


@app.route("/register", methods=["GET", "POST"])
def register():
    username = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")  # ✅ Hash the password
    new_user = User(username=username, email=email, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 200



@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()  # Flask-Login logout
    session.clear()  # Clear session data
    return jsonify({"message": "Logged out successfully"}), 200

# -------------------------------- Load AI Models Safely (Handle Missing Models)----------------------------------
try:
    print("🔄 Loading AI models...")
    cancer_classifier = TFAutoModelForImageClassification.from_pretrained("ShimaGh/Brain-Tumor-Detection")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    blood_model = joblib.load("models/blood_model.pkl") if os.path.exists("models/blood_model.pkl") else None
    urine_model = joblib.load("models/urine_model.pkl") if os.path.exists("models/urine_model.pkl") else None
    thyroid_model = joblib.load("models/thyroid_model.pkl") if os.path.exists("models/thyroid_model.pkl") else None

    genai.configure(api_key="AIzaSyBBCVqxouxx8YZoSSNiN_ngKA7fy2MEQGo")
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    
    
def extract_text_from_image(image_path):
    """Extract numerical values from medical reports using OCR."""
    image = cv2.imread(image_path)
    text = pytesseract.image_to_string(image)
    
    # Extract only numerical values
    extracted_values = [float(s) for s in re.findall(r'\d+\.\d+|\d+', text)]
    
    return extracted_values

@app.route("/analyze_file", methods=["POST"])
def analyze_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    uploaded_file.save(filepath)

    extracted_values = extract_text_from_image(filepath)
    
    # Send extracted values to backend for analysis
    if len(extracted_values) >= 12:
        blood_result = requests.post("http://127.0.0.1:5000/analyze_blood", json={"values": extracted_values[:12]}).json()
        return jsonify(blood_result)
    
    if len(extracted_values) >= 8:
        thyroid_result = requests.post("http://127.0.0.1:5000/analyze_thyroid", json={"values": extracted_values[:8]}).json()
        return jsonify(thyroid_result)
    
    return jsonify({"error": "Invalid report format."}), 400

def check_models():
    """Ensure all AI models are loaded before processing requests."""
    if not blood_model:
        return {"error": "Blood test model missing!"}
    if not urine_model:
        return {"error": "Urine test model missing!"}
    if not thyroid_model:
        return {"error": "Thyroid test model missing!"}
    return None

@app.route("/analyze_blood", methods=["POST"])
def analyze_blood():
    data = request.json
    values = data.get("values", [])
    if len(values) < 12:
        return jsonify({"error": "Invalid input data"}), 400

    result = analyze_blood_test(values)
    return jsonify({"Blood Test Prediction": result})


@app.route("/analyze_urine", methods=["POST"])
def analyze_urine():
    data = request.json
    values = data.get("values", [])
    
    if len(values) < 10:  # Ensure the correct number of features
        return jsonify({"error": "Invalid input data for urine test"}), 400

    result = analyze_urine_test(values)
    return jsonify({"Urine Test Prediction": result})


@app.route("/analyze_thyroid", methods=["POST"])
def analyze_thyroid():
    data = request.json
    values = data.get("values", [])
    if len(values) < 8:
        return jsonify({"error": "Invalid input data"}), 400

    result = analyze_thyroid_test(values)
    return jsonify({"Thyroid Test Prediction": result})



@app.route("/detect_cancer", methods=["POST"])
def detect_cancer():
    file = request.files["image"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Classify cancer type
    cancer_type, conf = classify_cancer_type(file_path)

    if conf < 50:
        return jsonify({"error": "Uncertain cancer classification. Please provide a clearer image."})

    # Select & apply specific detection model
    result = detect_cancer(file_path, cancer_type)

    return jsonify({"Cancer Type": cancer_type, "Result": result})



# ------------------- Chatbot Section------------------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"reply": "❌ Please enter a valid question."}), 400

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(query)
        return jsonify({"reply": format_response(response.text)})
    except Exception as e:
        return jsonify({"reply": "⚠️ AI service unavailable. Please try again later."}), 500

def format_response(response_text):
    """Formats AI-generated medical responses into short paragraphs and bullet points."""
    formatted_text = response_text.replace("**", "").replace("*", "")
    replacements = {
        "Tumors": "🩺 **Tumors Overview**\n",
        "Benign tumors": "\n😊 **Benign Tumors (Non-Cancerous)**\n",
        "Malignant tumors": "\n⚠️ **Malignant Tumors (Cancerous)**\n",
        "Causes": "\n🔬 **What Causes Tumors?**\n",
        "Symptoms": "\n🔍 **Common Symptoms of Tumors**\n",
        "Diagnosis": "\n🩺 **How Tumors Are Diagnosed**\n",
        "Treatment": "\n💊 **How Tumors Are Treated**\n",
        "Important Note": "\n⚠️ **Important Note**\n"
    }
    for key, value in replacements.items():
        formatted_text = formatted_text.replace(key, value)
    return formatted_text

# ----------------------------- Hospital Section------------------------------
@app.route("/geocode", methods=["GET"])
def geocode_location():
    location = request.args.get("location", "")
    geolocator = Nominatim(user_agent="hospital_locator")
    loc = geolocator.geocode(location)
    if loc:
        return jsonify({"lat": loc.latitude, "lon": loc.longitude})
    return jsonify({"error": "Location not found"}), 404



@app.route("/nearest_hospitals", methods=["POST"])
def find_hospitals():
    data = request.get_json()
    location = data.get("location", "")
    if re.match(r"^-?\d+\.\d+,-?\d+\.\d+$", location):
        lat, lon = map(float, location.split(","))
    else:
        geolocator = Nominatim(user_agent="hospital_locator")
        loc = geolocator.geocode(location)
        if loc:
            lat, lon = loc.latitude, loc.longitude
        else:
            return jsonify({"error": "Location not found"}), 404

    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""[out:json];node["amenity"="hospital"](around:5000,{lat},{lon});out;"""
    response = requests.get(overpass_url, params={"data": query}).json()

    hospitals = [{"name": node["tags"]["name"], "lat": node["lat"], "lon": node["lon"]} for node in response.get("elements", []) if "tags" in node and "name" in node["tags"]]
    return jsonify({"hospitals": hospitals[:5]}) if hospitals else jsonify({"error": "No hospitals found."})

# ----------------------Main-------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False)
