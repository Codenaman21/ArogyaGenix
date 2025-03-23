# 🏥 ArogyaGenix – AI-Powered Medical Diagnostics Platform

ArogyaGenix is an **AI-driven healthcare platform** designed to revolutionize medical diagnostics by leveraging **machine learning and deep learning models**. It enables **accurate disease detection** from medical reports and images, providing insightful analysis for both patients and healthcare professionals. 

---
## 🔍 Key Features

### **1. AI-Driven Medical Report Analysis** 📑
- Extracts text from **Blood, Urine, and Thyroid** reports using **OCR (Optical Character Recognition)**.
- Processes extracted values to perform **AI-based disease detection**.
- Displays comprehensive **diagnostic results with confidence scores**.

### **2. Multi-Cancer Detection** 🧬
- Detects multiple cancer types: **Lung, Brain, Breast, Skin, Prostate, and Oral Cancer**.
- Uses **state-of-the-art CNN and DenseNet models** for precision.
- Generates a **risk assessment severity bar** for better interpretation.

### **3. Intuitive Dashboard & User Experience** 📊
- Interactive UI for seamless **report uploads and result visualization**.
- Displays **disease name, confidence score, and severity level** in a structured manner.
- Provides a dropdown menu to choose between **Blood, Urine, and Thyroid** analysis.

### **4. AI-Powered Chatbot Integration** 🤖
- **Google Gemini AI-powered chatbot** for real-time health advice.
- Assists users with general medical queries based on AI knowledge.

### **5. Geolocation-Based Hospital Finder** 📍
- Identifies **nearest hospitals** based on user location.
- Offers navigation support for further medical assistance.

### **6. Secure Authentication System** 🔐
- Implements **Flask-Login & Flask-Bcrypt** for secure user authentication.
- Stores user credentials securely in **SQLite database**.

---
## ⚙️ System Workflow

1️⃣ **Upload Report/Image** – User uploads a medical document (PDF/Image).

2️⃣ **AI Analysis** – OCR extracts relevant data; ML models analyze health parameters.

3️⃣ **Diagnosis & Results** – System returns **disease name, confidence score, and severity level**.

4️⃣ **AI Chatbot Assistance** – Provides additional insights based on diagnosis.

5️⃣ **Hospital Locator** – Recommends medical facilities based on location.

---
## 🛠️ Technology Stack

### **Frontend (Web Interface)** 🎨
- **HTML, CSS, JavaScript** – UI/UX development.
- **Bootstrap** – Responsive design framework.

### **Backend (Server & API)** 🏗️
- **Flask (Python)** – Web framework for API & logic processing.
- **Flask-Login, Flask-Bcrypt** – Authentication & security.
- **SQLite** – Lightweight database management.

### **Machine Learning & AI Models** 🤖
- **TensorFlow, Keras, OpenCV** – AI-based image & text processing.
- **OCR (Tesseract)** – Text extraction from medical reports.
- **FineTuned Models & CNNs** – Advanced neural networks for multi-cancer detection.
- **Custom ML Models** – Trained for Blood, Urine, and Thyroid analysis.

### **APIs & Integrations** 🔗
- **Google Gemini AI** – Chatbot-powered medical insights.
- **Geolocation API** – Hospital locator functionality.

---
## 🚀 Installation Guide

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/ArogyaGenix.git
cd ArogyaGenix
```

### **2. Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Flask App**
```bash
python main1.py
```

### **5. Access the Web App** 🌐
Open your browser and navigate to:
👉 **http://127.0.0.1:5000/**

---
## 📷 Screenshots


---
## 📌 Future Enhancements

🚀 **Integration with Wearable Devices** – Real-time health monitoring.  
📊 **Advanced Data Visualization** – AI-powered health insights and risk predictions.  
📱 **Mobile App Development** – Bringing AI-driven diagnostics to smartphones.  

---
## 🤝 Contribution & Collaboration

We welcome contributions from developers, data scientists, and healthcare professionals! To contribute:
**1️⃣ Fork the repository.**

**2️⃣ Create a feature branch.**

**3️⃣ Commit changes and submit a Pull Request.**

For collaboration opportunities, feel free to reach out! 🚀

---
## 📜 License
This project is **open-source** under the **Apache2.0 License**.

---
## 📬 Contact
📧 **Email:** nsachdeva300@gmail.com  
🐙 **GitHub:** https://github.com/Codenaman21   

---

⭐ **If you find this project useful, consider giving it a star on GitHub!** ⭐

