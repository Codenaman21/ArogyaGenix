import os
import cv2
import numpy as np
import joblib
import pytesseract
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from transformers import ViTImageProcessor, TFAutoModelForImageClassification
from geopy.geocoders import Nominatim
import google.generativeai as genai  # Gemini chatbot
import logging

# Flask App Initialization
app = Flask(__name__ , template_folder="templates")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key_here"

# Ensure Required Folders Exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# Flask Extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        self.password = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password, password)
    
# ✅ Load User Function (PLACE IT HERE!)
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load AI Models
cancer_classifier = TFAutoModelForImageClassification.from_pretrained("ShimaGh/Brain-Tumor-Detection")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

blood_model = joblib.load("models/blood_model.pkl") if os.path.exists("models/blood_model.pkl") else None
urine_model = joblib.load("models/urine_model.pkl") if os.path.exists("models/urine_model.pkl") else None
thyroid_model = joblib.load("models/thyroid_model.pkl") if os.path.exists("models/thyroid_model.pkl") else None

genai.configure(api_key="YOUR_GEMINI_API_KEY")

# Routes
@app.route("/")
def index():
    return render_template("index3.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "error")
            return redirect(url_for("register"))
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful!", "success")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials. Try again.", "error")
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=current_user.username)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))

@app.route("/analyse", methods=["POST"])
def analyze_report():
    file = request.files["report"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    extracted_values = [float(s) for s in pytesseract.image_to_string(cv2.imread(file_path)).split() if s.replace(".", "", 1).isdigit()]
    
    if len(extracted_values) >= 8 and thyroid_model:
        thyroid_result = thyroid_model.predict([extracted_values[:8]])
        return jsonify({"Thyroid Prediction": thyroid_result.tolist()})
    elif len(extracted_values) >= 12 and blood_model:
        blood_result = blood_model.predict([extracted_values[:12]])
        return jsonify({"Blood Prediction": blood_result.tolist()})
    return "Invalid report format or missing model."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("message")
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(user_input)
    return jsonify({"response": response.text})

@app.route("/hospital-locator", methods=["POST"])
def find_nearest_hospital():
    data = request.get_json()
    location = data.get("location")
    geolocator = Nominatim(user_agent="hospital_locator")
    location_data = geolocator.geocode(location)
    if location_data:
        return jsonify({"latitude": location_data.latitude, "longitude": location_data.longitude})
    return jsonify({"error": "Location not found."})

# ✅ Hospital Locator Route
@app.route("/hospital")
def hospital():
    return render_template("hospital.html")  # ✅ Ensure "hospital.html" is in the "templates/" folder

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    print("🚀 Flask app running at: http://127.0.0.1:5000/")
    app.run(debug=True, use_reloader=False)
