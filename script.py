import google.generativeai as genai # gemini chatbot
import os
import cv2
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout  # Import Dropout
from tensorflow.keras.applications import EfficientNetB3  # Pre-trained model
from tensorflow.keras.applications import ResNet50 , VGG16 # Alternative Pre-trained model
from tensorflow.keras.applications import DenseNet121, MobileNetV2 , InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from transformers import ViTImageProcessor, TFAutoModelForImageClassification
from sklearn.ensemble import RandomForestClassifier #
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from twilio.rest import Client
from geopy.geocoders import Nominatim
import joblib
import re


genai.configure(api_key="AIzaSyADx7Qjws_DhV40lO7eeqHxsGFRKky1610")
TWILIO_ACCOUNT_SID = "your-twilio-sid"
TWILIO_AUTH_TOKEN = "your-twilio-auth-token"
print("Model Starting")

#-----------------------BLOOD, URINE & THYROID TEST ANALYSIS ----------------------------
def generate_medical_advice(prediction):
    advice = {
        "Anemia": "Increase iron-rich foods like spinach and red meat. Stay hydrated and consult a doctor for supplements.",
        "Diabetes": "Monitor blood sugar levels, reduce sugar intake, and maintain a balanced diet with regular exercise.",
        "Thyroid Issues": "Maintain a balanced iodine intake, get regular thyroid function tests, and follow prescribed medication.",
        "Kidney Disease": "Reduce salt and protein intake, stay hydrated, and consult a nephrologist for further advice.",
        "Lung Cancer": "Avoid smoking and exposure to pollutants. Follow prescribed treatments and consider a high-antioxidant diet.",
        "Brain Cancer": "Follow up with an oncologist for treatment options. Maintain a nutrient-rich diet and manage stress effectively.",
        "Breast Cancer": "Regular self-exams and follow-ups are essential. Maintain a healthy weight and consider lifestyle changes.",
        "Skin Cancer": "Avoid excessive sun exposure, use sunscreen, and get regular skin check-ups by a dermatologist.",
        "Prostate Cancer": "Follow dietary recommendations rich in vegetables. Consider regular screenings and lifestyle modifications.",
        "Colon Cancer": "Increase fiber intake, maintain a healthy diet, and get regular colonoscopies if recommended."
    }
    return advice.get(prediction, "To Know More use chatbot to Ask your Doubts and You Should Also consult a Doctor for Review.")


blood_df = pd.read_csv("datasets/blood_test.csv")


if "Gender" in blood_df.columns:
    blood_df["Gender"] = blood_df["Gender"].map({"M": 0, "F": 1}) 
    


X_blood, y_blood = blood_df.drop(columns=["Disease"]), blood_df["Disease"]


label_encoder = LabelEncoder()
y_blood = label_encoder.fit_transform(y_blood)  


X_train_blood, X_test_blood, y_train_blood, y_test_blood = train_test_split(X_blood, y_blood, test_size=0.2, random_state=42)


blood_model = RandomForestClassifier(n_estimators=100, random_state=42)
blood_model.fit(X_train_blood, y_train_blood)

print("Blood model trained successfully!")

joblib.dump(blood_model, "models/blood_model.pkl")

def validate_blood_test(values):
    """Ensures that blood test values are within realistic medical limits."""
    limits = {
        "Age": (0, 120),
        "Gender": (0, 1),  # 0 = Male, 1 = Female
        "Hemoglobin": (5, 20),  # g/dL
        "Platelet_Count": (100000, 450000),  
        "White_Blood_Cells": (2500, 15000),  
        "Red_Blood_Cells": (3.0, 6.5),  # million per µL
        "MCV": (70, 110),  # fL
        "MCH": (20, 40),  # pg
        "MCHC": (28, 38),  # g/dL
        "Glucose": (60, 250),  # mg/dL
        "Creatinine": (0.4, 2.5),  # mg/dL
        "TSH": (0.1, 10.0)  
    }

    
    for i, (key, (min_val, max_val)) in enumerate(limits.items()):
        if not (min_val <= values[i] <= max_val):
            raise ValueError(f"Invalid value for {key}: {values[i]} (must be between {min_val} and {max_val})")

    return True


def analyze_blood_test(values):
    
    
    validate_blood_test(values)
    
    
    feature_names = X_blood.columns.tolist()
    values_df = pd.DataFrame([values], columns=feature_names)
    
    
    probabilities = blood_model.predict_proba(values_df)[0]  
    
    
    max_prob_index = np.argmax(probabilities)  
    confidence = probabilities[max_prob_index] * 100  
    predicted_disease = label_encoder.inverse_transform([max_prob_index])[0]
    
    return f" **Most Likely Disease:** {predicted_disease} (Confidence: {confidence:.2f}%)"


print("Loading Urine Test Dataset...")
urine_df = pd.read_csv("datasets/urine_test.csv")


if "Gender" in urine_df.columns:
    urine_df["Gender"] = urine_df["Gender"].map({"M": 0, "F": 1})


categorical_cols = ["Protein", "Glucose", "Ketones", "Blood", "Leukocytes", "Nitrite"]
for col in categorical_cols:
    if col in urine_df.columns:
        urine_df[col] = urine_df[col].astype(str)  
        urine_df[col] = LabelEncoder().fit_transform(urine_df[col])  


X_urine, y_urine = urine_df.drop(columns=["Disease"]), urine_df["Disease"]


label_encoder_urine = LabelEncoder()
y_urine = label_encoder_urine.fit_transform(y_urine)


X_train_urine, X_test_urine, y_train_urine, y_test_urine = train_test_split(X_urine, y_urine, test_size=0.2, random_state=42)

print("Feature names used during training:", X_urine.columns.tolist())



urine_model = RandomForestClassifier(n_estimators=100, random_state=42)
urine_model.fit(X_train_urine, y_train_urine)
print(" Urine Test Model Trained Successfully!")

joblib.dump(urine_model, "models/urine_model.pkl")


def analyze_urine_test(values):
    
    
    feature_names = X_urine.columns.tolist()
    
    
    values_df = pd.DataFrame([values], columns=feature_names)
    
    
    probabilities = urine_model.predict_proba(values_df)[0]
    
    
    max_prob_index = np.argmax(probabilities)
    confidence = probabilities[max_prob_index] * 100
    predicted_disease = label_encoder_urine.inverse_transform([max_prob_index])[0]
    
    return f"Possible Disease: {predicted_disease} (Confidence: {confidence:.2f}%)"



print("Loading Thyroid Test Dataset...")

thyroid_df = pd.read_csv("datasets/thyroid_test.csv")


if "Gender" in thyroid_df.columns:
    thyroid_df["Gender"] = thyroid_df["Gender"].map({"M": 0, "F": 1})


X_thyroid, y_thyroid = thyroid_df.drop(columns=["Disease"]), thyroid_df["Disease"]


label_encoder_thyroid = LabelEncoder()
y_thyroid = label_encoder_thyroid.fit_transform(y_thyroid)  # Convert disease labels to numbers


smote = SMOTE(sampling_strategy="auto", random_state=42)
X_thyroid_balanced, y_thyroid_balanced = smote.fit_resample(X_thyroid, y_thyroid)

print("Original Dataset Shape:", X_thyroid.shape, "→ Balanced Dataset Shape:", X_thyroid_balanced.shape)


X_train_thyroid, X_test_thyroid, y_train_thyroid, y_test_thyroid = train_test_split(
    X_thyroid_balanced, y_thyroid_balanced, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_thyroid = scaler.fit_transform(X_train_thyroid)
X_test_thyroid = scaler.transform(X_test_thyroid)


thyroid_model = RandomForestClassifier(n_estimators=100, random_state=42)
thyroid_model.fit(X_train_thyroid, y_train_thyroid)

print(" Thyroid model trained successfully!")

joblib.dump(thyroid_model, "models/thyroid_model.pkl")

print(" Blood, Urine, and Thyroid models saved successfully!")

def analyze_thyroid_test(values):
    
   
    feature_names = X_thyroid.columns.tolist()
    

    values_df = pd.DataFrame([values], columns=feature_names)
    

    probabilities = thyroid_model.predict_proba(values_df)[0]  
    
   
    max_prob_index = np.argmax(probabilities)  
    confidence = probabilities[max_prob_index] * 100 
    predicted_disease = label_encoder_thyroid.inverse_transform([max_prob_index])[0] 
    
    return f" **Most Likely Thyroid Issue:** {predicted_disease} (Confidence: {confidence:.2f}%)"

#------------------------  MULTI-CANCER DETECTION (Deep Learning CNN)----------------------------


print(" Loading Vision Transformer for classification...")
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
image_classifier = TFAutoModelForImageClassification.from_pretrained("ShimaGh/Brain-Tumor-Detection")


cancer_classes = ["Brain Cancer", "Breast Cancer", "Lung Cancer", "Skin Cancer", "Prostate Cancer", "Oral Cancer"]


specialized_models = {
    "Brain Cancer": DenseNet121(weights="imagenet", include_top=True),
    "Breast Cancer": ResNet50(weights="imagenet", include_top=True),
    "Lung Cancer": VGG16(weights="imagenet", include_top=True),
    "Skin Cancer": EfficientNetB3(weights="imagenet", include_top=True),
    "Prostate Cancer": MobileNetV2(weights="imagenet", include_top=True),
    "Oral Cancer": InceptionV3(weights="imagenet", include_top=True),
}


preprocess_functions = {
    "Brain Cancer": densenet_preprocess,
    "Breast Cancer": resnet_preprocess,
    "Lung Cancer": vgg_preprocess,
    "Skin Cancer": efficientnet_preprocess,
    "Prostate Cancer": mobilenet_preprocess,
    "Oral Cancer": inception_preprocess,
}

print("All pre-trained models loaded successfully!")
specialized_models["Brain Cancer"].save("models/brain_cancer_model.h5")
specialized_models["Lung Cancer"].save("models/lung_cancer_model.h5")
specialized_models["Breast Cancer"].save("models/breast_cancer_model.h5")
specialized_models["Skin Cancer"].save("models/skin_cancer_model.h5")
specialized_models["Prostate Cancer"].save("models/Prostate_cancer_model.h5")
specialized_models["Oral Cancer"].save("models/Oral_cancer_model.h5")


# ----------------------- IMAGE CLASSIFICATION -----------------------

def classify_cancer_type(image_path):
    """Classifies the type of cancer using a Vision Transformer model."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    inputs = image_processor(images=image, return_tensors="tf")

   
    outputs = image_classifier(**inputs)
    probs = tf.nn.softmax(outputs.logits[0]).numpy()
    predicted_class = np.argmax(probs)

    confidence = probs[predicted_class] * 100

    return cancer_classes[predicted_class], confidence

# ------------------------- SPECIALIZED DETECTION MODELS -------------------------------
def detect_cancer(image_path, cancer_type):
    """Runs the specialized deep learning model for final analysis."""

    model = specialized_models.get(cancer_type)
    preprocess_input = preprocess_functions.get(cancer_type)

    if model is None:
        return "No specialized model found for this cancer type."

    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0).astype(np.float32)

   
    image = preprocess_input(image)

    
    predictions = model.predict(image)[0]
    max_index = np.argmax(predictions)
    confidence = predictions[max_index] * 100

    return f" **Detected:** {cancer_type} (Confidence: {confidence:.2f}%)"

# ----------------------FULL PIPELINE: CLASSIFY + DETECT --------------------------------

def full_cancer_detection(image_path):
    

    print(" Classifying cancer type...")
    cancer_type, conf = classify_cancer_type(image_path)

    if conf < 50:
        return " Uncertain classification. Please provide a clearer image."

    print(f"Cancer Type Identified: {cancer_type} (Confidence: {conf:.2f}%)")
    print(f" Running specialized detection model for {cancer_type}...")

    
    return detect_cancer(image_path, cancer_type)


# -----------------MEDICAL CHATBOT (Gemini-2 AI) -----------------------------------

def medical_chatbot(question):
    
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    temp = " you are a medical assistant bot and use this name only - arogya bot - answer concisely in around 100 words with key points - avoid jargon - use simple human language and answer this question"
    temp += " "
    temp += question
    response = model.generate_content(temp)

    
    formatted_response = format_response(response.text)

    return formatted_response


def format_response(response_text):
    

    
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

    
    formatted_text = formatted_text.replace("• ", "\n• ")

    
    formatted_text = re.sub(r"\n{2,}", "\n\n", formatted_text)  

    return formatted_text


# -------------------------- NEAREST HOSPITAL FINDER (Nomination Maps API) ---------------------------
def find_nearest_hospital(location):
    
    geolocator = Nominatim(user_agent="hospital_locator")
    location_data = geolocator.geocode(location)
    
    if location_data:
        latitude, longitude = location_data.latitude, location_data.longitude
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node
          ["amenity"="hospital"]
          (around:5000,{latitude},{longitude});
        out;
        """
        response = requests.get(overpass_url, params={"data": query}).json()
        
        hospitals = [element["tags"]["name"] for element in response.get("elements", []) if "name" in element.get("tags", {})]
        return hospitals[:5] if hospitals else ["No hospitals found nearby."]
    
    return ["Location not found."]
print("loaction okay")

# ----------------------- DOCTOR CONSULTATION VIA VIDEO CALL (Twilio API) ---------------------------
def create_twilio_video_call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    room = client.video.rooms.create(unique_name="MedicalConsultation")
    return f"Room created. Join with SID: {room.sid}"

# -------------------------  TESTING EXAMPLES -------------------------------------
if __name__ == "__main__":
    print(analyze_blood_test([30, 1, 13.5, 250000, 7000, 4.8, 90, 32, 34, 100, 1.2, 2.5]))  
    print(analyze_urine_test([30, 1, 2, 100, 0, 1, 0, 1, 1.02, 6.5])) 
    test_thyroid_values = [30, 1, 2.5, 1.8, 4.5, 30, 50, 1.2]  
    print(analyze_thyroid_test(test_thyroid_values)) 
    print(full_cancer_detection("test_cancer.jpg"))
    print(medical_chatbot("What are the symptoms of hairfall issues?"))
    print(find_nearest_hospital("28.659671, 77.091282"))