# 🏥 ArogyaGenix – AI-Powered Medical Diagnostics Suite

**ArogyaGenix** is an advanced AI-driven medical diagnostics platform that empowers healthcare through intelligent automation. Leveraging cutting-edge **machine learning**, **deep learning**, and **OCR technologies**, it delivers fast, accurate insights from both medical reports and scans — enabling patients and medical professionals to make informed decisions.

---

## 🚀 Key Highlights

### 🧠 AI-Powered Report Interpretation
- Extracts and processes data from **Blood**, **Urine**, and **Thyroid** reports using **OCR**.
- Performs intelligent diagnostics using **custom-trained ML models**.
- Presents comprehensive **diagnostic results** with **confidence levels** and **severity scores**.

### 🧬 Multi-Cancer Image Detection
- Detects **Lung**, **Brain**, **Breast**, **Skin**, **Prostate**, and **Oral** cancer types.
- Utilizes **CNNs and DenseNet** architectures for high-accuracy predictions.
- Visualizes **risk levels** via an intuitive **severity bar**.

### 🧾 Unified Diagnostic Dashboard
- User-friendly interface for uploading medical documents and viewing results.
- Displays **diagnosis**, **AI confidence**, and **severity level** in a clean, structured layout.
- Dropdown menu for selecting the type of test (Blood, Urine, Thyroid).

### 🤖 Integrated AI Chatbot
- Embedded **Google Gemini-powered chatbot** for contextual medical Q&A.
- Offers basic medical insights and first-level guidance based on user queries.

### 📍 Nearby Hospital Finder
- Auto-detects user location using **Geolocation API**.
- Recommends nearby hospitals and clinics, complete with navigation support.

### 🔐 Secure User Authentication
- Powered by **Flask-Login** and **Flask-Bcrypt** for secure, encrypted authentication.
- User credentials securely managed via **SQLite**.

---

## 🔀 Workflow Overview

1. **Upload Report/Image** – Users upload their medical document.
2. **AI-Based Processing** – OCR extracts data, which is analyzed by ML/DL models.
3. **Diagnosis Output** – Results are generated with confidence scores and severity indicators.
4. **Chatbot Interaction** – AI chatbot offers further clarification or suggestions.
5. **Hospital Recommendations** – Based on location, nearest hospitals are listed.

---

## ⚙️ Tech Stack Overview

### 🌐 Frontend
- **HTML, CSS, JavaScript** – Core UI development
- **Bootstrap** – Responsive and modern design framework

### 🧹 Backend
- **Python (Flask)** – Web server and API logic
- **SQLite** – Lightweight DB for user and report data
- **Flask-Login & Flask-Bcrypt** – Authentication and password hashing

### 🧠 AI & ML Models
- **TensorFlow & Keras** – Deep learning for image classification
- **OpenCV & Tesseract OCR** – Image preprocessing and text extraction
- **Custom ML models** – Trained for various pathology report types and cancer scans

### 🔗 Integrations
- **Google Gemini AI API** – AI chatbot integration
- **Geolocation API** – Location-aware hospital search

---

## 📦 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ArogyaGenix.git
cd ArogyaGenix
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
# Activate:
source venv/bin/activate     # For Linux/Mac
venv\Scripts\activate        # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App
```bash
python main1.py
```

### 5️⃣ Open the Web App
Navigate to:
👉 `http://127.0.0.1:5000/`

---

## 📁 Missing Files Notice

Due to GitHub size restrictions, some large files (e.g., model weights) have been excluded. To access them or set up a fully functional version, feel free to reach out:

📧 **Email:** nsachdeva300@gmail.com

---

## 🖼️ Preview Snapshots

### 🏠 Home Interface  
![Home](https://github.com/Codenaman21/ArogyaGenix/blob/main/images/Screenshot%202025-03-23%20103234.png)

### 📁 Report Analysis Dashboard  
![Dashboard](https://github.com/Codenaman21/ArogyaGenix/blob/main/images/Screenshot%202025-03-23%20103305.png)

### 🧬 Cancer Detection Result  
![Cancer Detection](https://github.com/Codenaman21/ArogyaGenix/blob/main/images/Screenshot%202025-03-23%20103335.png)

### 🤖 AI Chatbot Interaction  
![Chatbot](https://github.com/Codenaman21/ArogyaGenix/blob/main/images/Screenshot%202025-03-23%20103432.png)

### 📍 Hospital Locator  
![Locator](https://github.com/Codenaman21/ArogyaGenix/blob/main/images/Screenshot%202025-03-23%20103358.png)

> **Note:** If images aren’t loading, check the `/images/` directory in the repo.

---

## 🔮 Future Enhancements

- 📱 **Mobile App Version** – AI diagnostics on the go  
- 📈 **Visual Analytics** – Graphical trends from multiple test reports  
- 🧪 **Wearable Device Integration** – Real-time vitals + diagnostics  

---

## 🤝 Contribution Guidelines

We welcome enthusiastic developers, researchers, and medical professionals to contribute:

1. **Fork the repository**
2. **Create a new branch** for your feature/fix
3. **Push changes** and submit a **Pull Request**

### 👥 Team Members
- **Naman Sachdeva** – Team Leader
- **Chaitanya Dalal** – Contributor
- **Shivangi Gupta** – Contributor
- **Sanya Gautam** – Contributor

For collaborations or partnerships:  
📧 **Email:** nsachdeva300@gmail.com

---

## 📄 License

Licensed under the **Apache 2.0 License** – feel free to use, modify, and share responsibly.

---

## 🌟 Show Your Support

If ArogyaGenix made an impact or inspired you, consider giving it a ⭐️ on GitHub — your support means a lot!

