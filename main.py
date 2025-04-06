import os
import re
import cv2
import numpy as np
import joblib
import pytesseract
import torch
import pandas as pd
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import render_template, flash, redirect, url_for
from script import analyze_blood_test, analyze_urine_test, analyze_thyroid_test
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
import easyocr
import requests
from tensorflow.keras.models import load_model
from geopy.geocoders import Nominatim
import google.generativeai as genai  # Gemini chatbot
import logging

os.environ["PATH"] += os.pathsep + r"C:\Users\nsach\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR"

print("flight is taking off")

# Initialize Flask App
app = Flask(__name__, template_folder=os.path.abspath("templates") , static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # Add this line
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key_here"
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# -------------------------- User Authentication System --------------------------
print("flight is in 300km/hr speed")

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID."""
    return User.query.get(int(user_id))

@app.route("/register", methods=["POST"])
def register():
    """Register a new user."""
    username = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "User already exists"}), 400

    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 200

@app.route("/login", methods=["GET"])
def serve_login_page():
    return render_template("login.html")  # Ensure login.html is in the templates folder

@app.route("/login", methods=["POST"])
def login():
    """Handles user login."""
    try:
        data = request.form
        email = data.get("email")
        password = data.get("password")

        print("🔍 Received email:", email)  # Debugging Step
        print("🔍 Received password:", password)  # Debugging Step

        if not email or not password:
            print("❌ Missing login fields!")  # Debugging
            return jsonify({"error": "Email and password are required!"}), 400

        # Check if user exists in the database
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            print("❌ Invalid email or password!")  # Debugging
            return jsonify({"error": "Invalid email or password!"}), 401

        # Log in the user
        login_user(user, remember=True)
        print("✅ Login successful for:", user.email)  # Debugging

        return jsonify({
            "message": "Login successful!",
            "username": user.username,
            "email": user.email
        }), 200

    except Exception as e:
        print("❌ Error during login:", str(e))  # Debugging
        return jsonify({"error": "An error occurred: " + str(e)}), 500



@app.route("/logout", methods=["POST"])
@login_required
def logout():
    """Handles user logout properly."""
    logout_user()  # Logs out user from Flask-Login
    session.pop("user_id", None)  # Ensure session user is removed
    session.clear()  # Fully clear session data

    print("✅ User logged out!")  # Debugging
    return jsonify({"message": "Logout successful!", "redirect": "/login"}), 200


print("flight is in the air")


@app.route("/signup", methods=["POST"])
def signup():
    """Handles user registration."""
    try:
        data = request.form
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")

        print("🔍 Received Full Name:", username)  # Debugging Step
        print("🔍 Received email:", email)  # Debugging Step
        print("🔍 Received password:", password)  # Debugging Step

        if not username or not email or not password:
            print("❌ Missing signup fields!")  # Debugging
            return jsonify({"error": "All fields are required!"}), 400

        if len(password) < 6:
            print("❌ Password too short!")  # Debugging
            return jsonify({"error": "Password must be at least 6 characters long!"}), 400

        # Check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print("❌ Email already registered!")  # Debugging
            return jsonify({"error": "Email already in use!"}), 400

        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)  # Hash password

        db.session.add(new_user)
        db.session.commit()

         # 🔑 Log in the user after successful signup
        login_user(new_user, remember=True)  # Add this line
        
        print("✅ Signup successful for:", email)  # Debugging
        return jsonify({"message": "Signup successful!"}), 201

    except Exception as e:
        print("❌ Error during signup:", str(e))  # Debugging
        return jsonify({"error": "An error occurred: " + str(e)}), 500
    
    
@app.route("/signin", methods=["GET", "POST"])
def signin():
    """Handles user login (Sign In)."""
    if request.method == "GET":
        print("🔍 Redirecting GET request for /signin")
        return redirect("/")  # Redirects to home page instead of error

    try:
        data = request.form
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required!"}), 400

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({"error": "Invalid email or password!"}), 401

        login_user(user, remember=True)
        return jsonify({
            "message": "Sign In successful!",
            "username": user.username,
            "email": user.email
        }), 200

    except Exception as e:
        return jsonify({"error": "An error occurred: " + str(e)}), 500

    

# --------------------------------Model access and Processes----------------------------------

# Ensure upload folder exists
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ✅ Load AI models
try:
    print("🔄 Loading AI models...")

    def safe_load_model(path, model_type="pkl"):
        """Safely loads models to avoid crashes if a model file is missing."""
        if os.path.exists(path):
            return joblib.load(path) if model_type == "pkl" else load_model(path)
        else:
            print(f"⚠️ Warning: Model {path} not found. Skipping...")
            return None

    # Load models for textual analysis (Blood, Urine, Thyroid)
    blood_model = safe_load_model("models/blood_model.pkl", "pkl")
    urine_model = safe_load_model("models/urine_model.pkl", "pkl")
    thyroid_model = safe_load_model("models/thyroid_model.pkl", "pkl")

    # Load individual cancer detection models
    brain_cancer_model = safe_load_model("models/brain_cancer_model.h5", "h5")
    breast_cancer_model = safe_load_model("models/breast_cancer_model.h5", "h5")
    lung_cancer_model = safe_load_model("models/lung_cancer_model.h5", "h5")
    skin_cancer_model = safe_load_model("models/skin_cancer_model.h5", "h5")
    prostate_cancer_model = safe_load_model("models/prostate_cancer_model.h5", "h5")
    oral_cancer_model = safe_load_model("models/oral_cancer_model.h5", "h5")

    # ✅ Load Hugging Face model for cancer classification
    print("🔍 Loading 'ErnestBeckham/MulticancerViT' model for cancer type classification...")
    hf_model = AutoModelForImageClassification.from_pretrained("VyChau/cancer-detect")
    hf_processor = AutoProcessor.from_pretrained("VyChau/cancer-detect")

    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ✅ Map cancer type to deep learning models
cancer_models = {
    "brain": brain_cancer_model,
    "breast": breast_cancer_model,
    "lung": lung_cancer_model,
    "skin": skin_cancer_model,
    "prostate": prostate_cancer_model,
    "oral": oral_cancer_model
}

# ✅ Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

POPPLER_PATH = r"C:\poppler\poppler-24.08.0\Library\bin"

# ✅ Function to extract text from reports (Images & PDFs)
def extract_text_from_report(file_path):
    """Extracts text from images or PDFs."""
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        text = "\n".join([pytesseract.image_to_string(img, config='--psm 6') for img in images]).strip()
    else:
        img = cv2.imread(file_path)
        if img is None:
            return None  # Instead of empty string, return None for better error handling
        text = pytesseract.image_to_string(img, config='--psm 6').strip()

    if len(text) < 10:  # If extracted text is too short, return an error
        return None

    return text


# ✅ Function to extract numerical values from medical text
def extract_medical_values(text):
    """Extract numerical values from medical report text using regex."""
    patterns = {
        "Hemoglobin": r"Hemoglobin.*?([\d.]+)",
        "RBC": r"Total RBC count.*?([\d.]+)",
        "PCV": r"Packed Cell Volume.*?([\d.]+)",
        "MCV": r"Mean Corpuscular Volume.*?([\d.]+)",
        "MCH": r"MCH\s+([\d.]+)",
        "MCHC": r"MCHC\s+([\d.]+)",
        "RDW": r"RDW\s+([\d.]+)",
        "WBC": r"Total WBC count.*?([\d,]+)",
        "Neutrophils": r"Neutrophils\s+([\d.]+)",
        "Lymphocytes": r"Lymphocytes\s+([\d.]+)",
        "Eosinophils": r"Eosinophils\s+([\d.]+)",
        "Platelet_Count": r"Platelet Count.*?([\d,]+)"
    }

    extracted_values = {key: float(re.search(pattern, text).group(1).replace(",", "")) if re.search(pattern, text) else 0.0 for key, pattern in patterns.items()}
    return extracted_values

# ✅ Function to analyze blood reports using ML models
def analyze_blood_test(values):
    """Predicts blood test results using ML model."""
    try:
        values_array = np.array(list(values.values())).reshape(1, -1)
        prediction = blood_model.predict(values_array)

        class_labels = {
            0: "Normal",
            1: "Mild Anemia",
            2: "Severe Anemia",
            3: "Low White Blood Cell Count",
            4: "High Platelet Count",
            5: "Thyroid Imbalance",
            6: "Diabetes Risk",
            7: "Kidney Dysfunction"
        }
        return class_labels.get(prediction[0], "Unknown Condition")
    except Exception as e:
        return {"error": str(e)}

# ✅ Upload Blood/Urine/Thyroid Report (TEXTUAL)
@app.route("/upload_report_textual", methods=["POST"])
def upload_report_textual():
    """Processes medical reports (Blood, Urine, Thyroid) from images or PDFs."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    extracted_text = extract_text_from_report(file_path)
    if extracted_text is None:
        return jsonify({"error": "Could not extract text from the report. Ensure it's a clear scan."}), 400

    medical_values = extract_medical_values(extracted_text)
    if sum(medical_values.values()) == 0:
        return jsonify({"error": "No valid medical data extracted"}), 400

    return jsonify({"extracted_data": medical_values, "diagnosis_results": analyze_blood_test(medical_values)})


# ✅ Cancer Type Classification
def classify_cancer_type(image_path):
    """Uses Hugging Face's ViT model to classify cancer type from MRI/CT/X-ray images."""
    img = Image.open(image_path).convert("RGB")
    inputs = hf_processor(images=img, return_tensors="pt")

    with torch.no_grad():
        outputs = hf_model(**inputs)
        predicted_class = outputs.logits.argmax().item()

    class_map = {0: "brain", 1: "breast", 2: "lung", 3: "skin", 4: "prostate", 5: "oral"}
    return class_map.get(predicted_class, "unknown")


def process_cancer_image(image_path, model, cancer_type):
    """Processes MRI, CT, and X-ray images for cancer detection."""
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Unable to process the image. Please upload a valid medical image."})

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0).astype("float32") / 255.0  # Ensure float32 type for model input

    prediction = model.predict(img)
    result = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer"
    
    return jsonify({"result": result, "cancer_type": cancer_type})


# ✅ Upload Cancer Detection Image
@app.route("/upload_report_cancer", methods=["POST"])
def upload_report_cancer():
    """Automatically detects cancer type and processes image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    cancer_type = classify_cancer_type(file_path)
    return process_cancer_image(file_path, cancer_models.get(cancer_type), cancer_type)

# ------------------------- Chatbot, nearest Hospital , and routes to html-------------------------------------
print("wapis seatbelt pahniye")
#  Routes
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


@app.route("/dashboard-data", methods=["GET"])
@login_required  # ✅ Prevents unauthorized users from accessing dashboard
def dashboard_data():
    """Fetch user details for the dashboard."""
    if not current_user.is_authenticated:
        return jsonify({"error": "Unauthorized"}), 401  # Ensure user is logged in

    return jsonify({
        "username": current_user.username,
        "email": current_user.email
    }), 200



print("flight land hone wali hai")

# ------------------- Chatbot Section------------------------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"reply": " Please enter a valid question."}), 400

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(query)
        return jsonify({"reply": format_response(response.text)})
    except Exception as e:
        return jsonify({"reply": " AI service unavailable. Please try again later."}), 500

def format_response(response_text):
    """Formats AI-generated medical responses into short paragraphs and bullet points."""
    formatted_text = response_text.replace("**", "").replace("*", "")
    replacements = {
        "Tumors": " **Tumors Overview**\n",
        "Benign tumors": "\n **Benign Tumors (Non-Cancerous)**\n",
        "Malignant tumors": "\n **Malignant Tumors (Cancerous)**\n",
        "Causes": "\n **What Causes Tumors?**\n",
        "Symptoms": "\n **Common Symptoms of Tumors**\n",
        "Diagnosis": "\n **How Tumors Are Diagnosed**\n",
        "Treatment": "\n **How Tumors Are Treated**\n",
        "Important Note": "\n **Important Note**\n"
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

print("flight land ho gayi ha")
# ----------------------Main-------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False)
