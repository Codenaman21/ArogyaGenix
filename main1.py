import os
import re
import cv2
print("sone0")
import numpy as np
import joblib
import pytesseract
import pandas as pd
print("heloo")
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import render_template, flash, redirect, url_for
#from script import analyze_blood_test, analyze_urine_test, analyze_thyroid_test
#from transformers import AutoModelForImageClassification, AutoProcessor
from pdf2image import convert_from_path
from werkzeug.utils import secure_filename
print("dance")
import easyocr
print("run")
import torch
print("jump")
from PIL import Image
import requests
print("done")
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet121
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import ViTImageProcessor, TFAutoModelForImageClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
print("done2")
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from geopy.geocoders import Nominatim
import google.generativeai as genai  # Gemini chatbot
import logging
import traceback


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("flight is taking off")


app = Flask(__name__, template_folder=os.path.abspath("templates") , static_folder="static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax" 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "AIzaSyDPx4TC9slcz2OLnEC4fbIb6aAsxLghJzY"
app.config["UPLOAD_FOLDER"] = "uploads"  
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

def get_user_from_db(user_id):
    from your_database_model import User  # Replace with your actual model
    user = User.query.filter_by(id=user_id).first()
    return user
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
    return render_template("login.html")  

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    """Handles user login."""
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
            "message": "Login successful!",
            "username": user.username,
            "email": user.email
        }), 200

    except Exception as e:
        return jsonify({"error": "An error occurred: " + str(e)}), 500

@app.route("/signin", methods=["POST"])
def signin():
    """Redirects sign-in requests to login."""
    return login()




@app.route("/logout", methods=["POST"])
@login_required
def logout():
    
    session.pop("user_id", None)  
    session.clear() 

    print(" User logged out!") 
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

        print(" Received Full Name:", username) 
        print(" Received email:", email)  
        print(" Received password:", password)  
        if not username or not email or not password:
            print(" Missing signup fields!") 
            return jsonify({"error": "All fields are required!"}), 400

        if len(password) < 6:
            print(" Password too short!")  
            return jsonify({"error": "Password must be at least 6 characters long!"}), 400

       
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print(" Email already registered!") 
            return jsonify({"error": "Email already in use!"}), 400

        
        new_user = User(username=username, email=email)
        new_user.set_password(password)  

        db.session.add(new_user)
        db.session.commit()

        
        login_user(new_user, remember=True)  
        
        print(" Signup successful for:", email) 
        return jsonify({"message": "Signup successful!"}), 201

    except Exception as e:
        print(" Error during signup:", str(e))  
        return jsonify({"error": "An error occurred: " + str(e)}), 500
    
    

# --------------------------------Model access and Processes----------------------------------

# Ensure upload folder exists
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

try:
    print(" Loading AI models...")

    def safe_load_model(path, model_type="pkl"):
        if os.path.exists(path):
            return joblib.load(path) if model_type == "pkl" else load_model(path)
        else:
            print(f" Warning: Model {path} not found. Skipping...")
            return None

    blood_model = safe_load_model("models/blood_model.pkl", "pkl")
    urine_model = safe_load_model("models/urine_model.pkl", "pkl")
    thyroid_model = safe_load_model("models/thyroid_model.pkl", "pkl")

    brain_cancer_model = safe_load_model("models/brain_cancer_model.h5", "h5")
    breast_cancer_model = safe_load_model("models/breast_cancer_model.h5", "h5")
    lung_cancer_model = safe_load_model("models/lung_cancer_model.h5", "h5")
    skin_cancer_model = safe_load_model("models/skin_cancer_model.h5", "h5")
    prostate_cancer_model = safe_load_model("models/prostate_cancer_model.h5", "h5")
    oral_cancer_model = safe_load_model("models/oral_cancer_model.h5", "h5")

    print(" Loading ViT for cancer classification...")
    try:
        vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_model = AutoModelForImageClassification.from_pretrained("ShimaGh/Brain-Tumor-Detection")  # Ensure correct model
        print(" ViT Model Label Mapping:", vit_model.config.id2label)
        print(" ViT Model Loaded Successfully!")
    except Exception as e:
        print(f" Error loading ViT Model: {e}")
        vit_model = None  # Prevent crashes

except Exception as e:
    print(f" Error loading models: {e}")

print("seat belt khol sakte hai")

ocr_reader = easyocr.Reader(['en'])


model_map = {
    "brain": brain_cancer_model,
    "breast": breast_cancer_model,
    "lung": lung_cancer_model,
    "skin": skin_cancer_model,
    "prostate": prostate_cancer_model,
    "oral": oral_cancer_model
}



ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf"}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

POPPLER_PATH = r"C:\Users\nsach\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

os.environ["PATH"] += os.pathsep + POPPLER_PATH

def classify_cancer_type(image_path):
    global vit_model, vit_processor
    
    
    image = Image.open(image_path).convert("RGB")
    inputs = vit_processor(images=image, return_tensors="pt")
    
    
    outputs = vit_model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    confidence_score = torch.softmax(outputs.logits, dim=-1)[0][predicted_class_idx].item() * 100

    
    manual_class_map = {
        0: "brain", 
        1: "breast",
        2: "lung",
        3: "skin",
        4: "prostate",
        5: "oral"
    }
    
   
    detected_label = manual_class_map.get(predicted_class_idx, "unknown")
    
    print(f" Classified Cancer Type: {detected_label} (Confidence: {confidence_score:.2f}%)")
    
    return detected_label, confidence_score




def extract_text_from_image(file_path):
    """Extract text from medical reports. Supports both images and PDFs."""
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path) 
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, config='--psm 6') + "\n"
        print("\n Extracted Text from PDF:\n", text)  
        return text.strip()  

    # Handle image files (JPG, PNG, etc.)
    img = cv2.imread(file_path)
    if img is None:
        return "Error: Could not read image file. Ensure it's a valid format."

    extracted_text = pytesseract.image_to_string(img, config='--psm 6')
    print("\n Extracted Text from Image:\n", extracted_text)  
    return extracted_text




def clean_text(text):
    """Normalize extracted text to fix OCR distortions before regex matching."""
    text = text.lower()
    
    
    replacements = {
        "cumm": "/mm³", "mill/cumm": "10^6/mm³", "million/cumm": "10^6/mm³",
        "lakhs/cumm": "00000/mm³",  
        "10°6/l": "10^6/l", "10°3/uL": "10^3/uL", "pg": "Pg",
        "vise: \\ieasure": "hematocrit",  
        "pednco)": "platelet count", 
        "gid.": "g/dL", 
        "flow cyomery": "flow cytometry",  
        "calcuiavew": "calculated",  
        "caleviaied": "calculated",
        "cateuiaved": "calculated",
        "lo ": "", 
        "to ": "",  
        "hct": "hematocrit",  
        "mchc h ": "mchc ",  
        "rbc count inypadance": "rbc count",  
        "platelet count platelet count": "platelet count", 
        "mcv calculated": "mcv", 
    }

    for key, value in replacements.items():
        text = text.replace(key, value)

   
    text = re.sub(r'[^a-zA-Z0-9.%/\- ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_medical_values(text):
    text = clean_text(text)  
    print("\n Cleaned Extracted Text for Debugging:\n", text)  

   
    patterns = {
        "Hemoglobin": r"hemoglobin\s*\(?hb\)?\s*([\d.]+)",  
        "RBC": r"total rbc count\s*([\d.]+)", 
        "PCV": r"packed cell volume\s*\(?pcv\)?\s*([\d.]+)",  
        "MCV": r"mean corpuscular volume\s*\(?mcv\)?\s*([\d.]+)",  
        "MCH": r"mch\s*([\d.]+)",  
        "MCHC": r"mchc\s*([\d.]+)", 
        "RDW": r"rdw\s*([\d.]+)",  
        "WBC": r"total wbc count\s*([\d,]+)", 
        "Neutrophils": r"neutrophils\s*([\d.]+)", 
        "Lymphocytes": r"lymphocytes\s*([\d.]+)",  
        "Eosinophils": r"eosinophils\s*([\d.]+)",  
        "Platelet_Count": r"platelet count\s*([\d,]+)"  
    }

    extracted_values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace(",", "")  
            try:
                extracted_values[key] = float(value)
            except ValueError:
                print(f" [WARNING] Could not convert {key}: {value}")
                extracted_values[key] = 0.0  
        else:
            extracted_values[key] = 0.0

    print("\n Extracted Medical Values:", extracted_values)  
    return extracted_values



def analyze_medical_data(values, model):
    """Predict disease based on extracted values using ML models."""
    input_data = np.array([list(values.values())])
    prediction = model.predict(input_data)
    return {"Diagnosis": prediction.tolist()}


@app.route("/upload_report", methods=["POST"])
def process_report():
    """Process uploaded medical reports. Determines whether to analyze blood tests or detect cancer."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

   
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        print(" Detected an image. Checking if it's a cancer scan or a textual medical report...")

        extracted_text = extract_text_from_image(file_path)

        if len(extracted_text.strip()) > 20: 
            print(" Treating as a textual medical report (Blood/Urine/Thyroid)")
            medical_values = extract_medical_values(extracted_text)
            
            if sum(medical_values.values()) == 0:
                return jsonify({"error": "No valid medical data extracted"}), 400

            return jsonify({
                "extracted_data": medical_values,
                "diagnosis_results": analyze_blood_test(medical_values)
            })

        else: 
            print(" Treating as a cancer detection image.")
            return process_cancer_image(file_path)

    elif filename.lower().endswith(".pdf"): 
        extracted_text = extract_text_from_image(file_path)
        medical_values = extract_medical_values(extracted_text)
        if sum(medical_values.values()) == 0:
            return jsonify({"error": "No valid medical data extracted"}), 400

        return jsonify({
            "extracted_data": medical_values,
            "diagnosis_results": analyze_blood_test(medical_values)
        })

    return jsonify({"error": "Unsupported file format"}), 400



def analyze_blood_test(values):
    """Predicts blood test results using the trained ML model."""
    try:
        feature_names = ["Hemoglobin", "RBC", "PCV", "MCV", "MCH", "MCHC", "RDW", "WBC", "Neutrophils", "Lymphocytes", "Eosinophils", "Platelet_Count"]

        
        values_list = [values.get(key, 0.0) for key in feature_names]
        values_array = np.array(values_list).reshape(1, -1)

        
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

        
        predicted_class = prediction[0]
        diagnosis_label = class_labels.get(predicted_class, "Unknown Condition")

        
        confidence_score = float(np.max(blood_model.predict_proba(values_array))) * 100

        return {
            "diagnosis": diagnosis_label,
            "confidence_score": confidence_score
        }

    except Exception as e:
        print("\n Error in Model Prediction:", str(e))
        return {"error": str(e)}
    
    
def analyze_urine_test(values):
    """Predicts urine test results using the trained ML model."""
    try:
        feature_names = [
            "Glucose", "Protein", "pH", "Specific_Gravity", 
            "Ketones", "Bilirubin", "Urobilinogen", "Nitrites", 
            "Leukocytes", "Blood"
        ]

        
        values_list = [values.get(key, 0.0) for key in feature_names]
        values_array = np.array(values_list).reshape(1, -1)

        
        prediction = urine_model.predict(values_array)

        
        class_labels = {
            0: "Normal",
            1: "Possible Urinary Tract Infection (UTI)",
            2: "Kidney Dysfunction",
            3: "Diabetes Risk",
            4: "Dehydration",
            5: "Bacterial Infection",
            6: "Acidosis",
            7: "Alkalosis",
            8: "High Ketones",
            9: "High Protein",
            10: "High Glucose",
            11: "Blood in Urine",
            12:"Leukocytes in Urine",
            13: "Nitrite Positive"
        }

        
        predicted_class = prediction[0]
        diagnosis_label = class_labels.get(predicted_class, "Unknown Condition")

        
        confidence_score = float(np.max(urine_model.predict_proba(values_array))) * 100

        return {
            "diagnosis": diagnosis_label,
            "confidence_score": confidence_score
        }

    except Exception as e:
        print("\n Error in Model Prediction:", str(e))
        return {"error": str(e)}


@app.route('/analyze_thyroid', methods=['POST'])
def analyze_thyroid():
    try:
        print("\n [INFO] Request received at /analyze_thyroid")
        print(f" Request Headers: {request.headers}")
        print(f" Request Files: {request.files}")

        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        
        extracted_text = extract_text_from_image(file_path)  # This function handles both image & PDF
        if not extracted_text.strip():
            return jsonify({"error": "Failed to extract text from file"}), 400

        
        medical_values = extract_medical_values_thyroid(extracted_text)
        if sum(medical_values.values()) == 0:
            return jsonify({"error": "No valid medical data extracted"}), 400

       
        result = analyze_thyroid_test(medical_values)

        
        return jsonify(result)

    except Exception as e:
        print("\n ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

def extract_medical_values_thyroid(text):
    text = clean_text(text) 

    patterns = {
        "TSH": r"(?i)tsh\s*[:\-]?\s*([\d.]+)",
        "T3": r"(?i)t3\s*[:\-]?\s*([\d.]+)",
        "T4": r"(?i)t4\s*[:\-]?\s*([\d.]+)",
        "FTI": r"(?i)free thyroxine index\s*[:\-]?\s*([\d.]+)",
        "TPOAb": r"(?i)thyroid peroxidase antibodies\s*[:\-]?\s*([\d.]+)",
        "TgAb": r"(?i)thyroglobulin antibodies\s*[:\-]?\s*([\d.]+)",
        "TSH_Receptor_Ab": r"(?i)tsh receptor antibodies\s*[:\-]?\s*([\d.]+)",
        "Age": r"(?i)age\s*[:\-]?\s*([\d.]+)" 
    }

    extracted_values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            try:
                extracted_values[key] = float(value)
            except ValueError:
                extracted_values[key] = 0.0  
        else:
            extracted_values[key] = 0.0  

    print("\n Extracted Thyroid Values:", extracted_values)
    return extracted_values

def analyze_thyroid_test(values):
    try:
        feature_names = [
            "TSH", "T3", "T4", "FTI", "TPOAb", "TgAb", "TSH_Receptor_Ab", "Age"
        ]

        
        values_list = [values.get(key, 0.0) for key in feature_names]
        values_array = np.array(values_list).reshape(1, -1)

        
        prediction = thyroid_model.predict(values_array)

        
        class_labels = {
            0: "Euthyroid (Normal Thyroid Function)",
            1: "Hypothyroidism",
            2: "Hyperthyroidism",
            3: "Hashimoto's Thyroiditis",
            4: "Graves' Disease",
            5: "Thyroid Cancer",
            6: "Subclinical Hypothyroidism",
            7: "Subclinical Hyperthyroidism"
        }

       
        predicted_class = prediction[0]
        diagnosis_label = class_labels.get(predicted_class, "Unknown Condition")

        
        confidence_score = float(np.max(thyroid_model.predict_proba(values_array))) * 100

        return {
            "diagnosis": diagnosis_label,
            "confidence_score": confidence_score
        }

    except Exception as e:
        print("\n Error in Model Prediction:", str(e))
        return {"error": str(e)}
    

@app.route('/analyze_blood', methods=['POST'])
def analyze_blood():
    try:
        print("\n [INFO] Request received at /analyze_blood")
        print(f" Request Headers: {request.headers}")
        print(f" Request Files: {request.files}")

       
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        
        extracted_text = extract_text_from_image(file_path) 
        if not extracted_text.strip():
            return jsonify({"error": "Failed to extract text from file"}), 400

        
        medical_values = extract_medical_values(extracted_text)
        if sum(medical_values.values()) == 0:
            return jsonify({"error": "No valid medical data extracted"}), 400

        
        result = analyze_blood_test(medical_values)

        
        return jsonify(result)

    except Exception as e:
        print("\n ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
    
    
@app.route('/analyze_urine', methods=['POST'])
def analyze_urine():
    try:
        print("\n [INFO] Request received at /analyze_urine")
        print(f" Request Headers: {request.headers}")
        print(f" Request Files: {request.files}")

        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

       
        extracted_text = extract_text_from_image(file_path) 
        if not extracted_text.strip():
            return jsonify({"error": "Failed to extract text from file"}), 400

        
        medical_values = extract_medical_values_urine(extracted_text)
        if sum(medical_values.values()) == 0:
            return jsonify({"error": "No valid medical data extracted"}), 400

        
        result = analyze_urine_test(medical_values)

        
        return jsonify(result)

    except Exception as e:
        print("\n ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
    
    
def extract_medical_values_urine(text):
    """Extract numerical values from urine test report text."""
    text = clean_text(text)  

    patterns = {
        "Glucose": r"(?i)glucose.*?(neg|nil|[\d.]+)",
        "Protein": r"(?i)protein.*?(neg|nil|[\d.]+)",
        "pH": r"(?i)ph\s*[:\-]?\s*([\d.]+)",
        "Specific_Gravity": r"(?i)specific gravity\s*[:\-]?\s*([\d.]+)",
        "Ketones": r"(?i)ketones.*?(neg|nil|[\d.]+)",
        "Bilirubin": r"(?i)bilirubin.*?(neg|nil|[\d.]+)",
        "Urobilinogen": r"(?i)urobilinogen.*?(normal|[\d.]+)",
        "Nitrites": r"(?i)nitrite.*?(neg|nil|[\d.]+)",
        "Leukocytes": r"(?i)white blood cells.*?([\d.]+)",
        "Blood": r"(?i)blood.*?(neg|nil|[\d.]+)"
    }

    extracted_values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if value.lower() in ["neg", "nil", "normal"]: 
                extracted_values[key] = 0.0
            else:
                try:
                    extracted_values[key] = float(value)
                except ValueError:
                    extracted_values[key] = 0.0 
        else:
            extracted_values[key] = 0.0  

    print("\n Extracted Urine Values:", extracted_values)
    return extracted_values



print("khanna kha li jiye")
def process_cancer_image(image_path, model, cancer_type):
    """Detects cancer using the specialized deep learning model."""
    
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 

    prediction = model.predict(img_array)
    confidence_score = float(prediction[0][0])  

    result = "Cancer Detected" if confidence_score > 0.45 else "No Cancer"

    return jsonify({
        "result": result,
        "cancer_type": cancer_type,
        "confidence": confidence_score
    })



@app.route("/upload_cancer_image", methods=["POST"])
def upload_cancer_image():
    """Detects cancer type using ViT and processes image using the specialized model."""

    print("\n [INFO] Request received at /upload_cancer_image")
    print(f" Request Headers: {request.headers}")
    print(f" Request Files: {request.files}")

    if "file" not in request.files:
        print(" [ERROR] No file received!")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        print(" [ERROR] Empty file name!")
        return jsonify({"error": "No selected file"}), 400

    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    print(f" [INFO] Saving file to: {file_path}")
    file.save(file_path)

    
    if vit_model is None:
        print(" [ERROR] ViT model not loaded!")
        return jsonify({"error": "Cancer classification model not loaded."}), 500

    
    print(f" [INFO] Classifying Cancer Type using ViT for {filename}...")
    detected_cancer_type, confidence_score = classify_cancer_type(file_path)
    print(f" [RESULT] Detected Cancer Type: {detected_cancer_type} (Confidence: {confidence_score:.2f}%)")

    
    cancer_full_names = {
        "breast": "Breast Cancer",
        "lung": "Lung Cancer",
        "brain": "Brain Cancer",
        "skin": "Skin Cancer",
        "prostate": "Prostate Cancer",
        "oral": "Oral Cancer",
    }

    
    full_cancer_name = cancer_full_names.get(detected_cancer_type, "Unknown Cancer")

    if full_cancer_name == "Unknown Cancer":
        print(f" [ERROR] Invalid Cancer Type Detected: {detected_cancer_type}")
        return jsonify({"error": "Invalid cancer type detected."}), 400

    
    response_text = f"**Detected:** {full_cancer_name} (Confidence: {confidence_score:.2f}%)"
    
    print(f" [SUCCESS] Returning: {response_text}")
    return jsonify({"message": response_text}), 200




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

@app.route("/analyse1")
def analyse1():
    return render_template("analyse1.html")

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


@app.route('/dashboard-data')
def dashboard_data():
    if 'user_id' in session:
        user = get_user_from_db(session['user_id'])
        return jsonify({"username": user.username, "email": user.email})
    return jsonify({"error": "Unauthorized"}), 401



print("flight land hone wali hai")

# ------------------- Chatbot Section------------------------
genai.configure(api_key="AIzaSyDPx4TC9slcz2OLnEC4fbIb6aAsxLghJzY")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"reply": " Please enter a valid question."}), 400

    try:
        print(f" [DEBUG] Received query: {query}")  
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(query)
        print(f" [DEBUG] AI response: {response.text}")  
        return jsonify({"reply": format_response(response.text)})
    except Exception as e:
        print(f" [ERROR] Chatbot error: {str(e)}")
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


#------------- deasises summary ------------------------
SUMMARY_API_KEY = "AIzaSyC4PUicuodAf-0Yrtoe_2Ud40DNElK9yUY"

def get_summary_model():
    genai.configure(api_key=SUMMARY_API_KEY)
    return genai.GenerativeModel("gemini-1.5-pro-latest")

summary_model = get_summary_model()

@app.route("/get_disease_summary", methods=["POST"])
def get_disease_summary():
    try:
        data = request.get_json(force=True)
        print("Received JSON:", data)

        disease = data.get("disease")
        if not disease:
            return jsonify({"error": "Disease name not provided"}), 400

        print(f" Received disease: {disease}")

        prompt = f"Explain the disease '{disease}' in very simple language within 80 words."
        response = summary_model.generate_content(prompt)

        print(" Gemini response:", response.text.strip())
        return jsonify({"summary": response.text.strip()})

    except Exception as e:
        print(" Exception in /get_disease_summary:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500
# ----------------------Main-------------------------------

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False)