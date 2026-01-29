# 🧠 Parkinson’s Detection System

An AI-powered full-stack web application for early Parkinson’s Disease detection using spiral drawing analysis and voice signal processing.

---

## 📌 Overview

Parkinson’s Disease is a progressive neurological disorder affecting motor skills and speech. Early detection significantly improves treatment outcomes.

This system combines:

- 🌀 CNN-based spiral image analysis  
- 🎙️ Voice signal feature classification  
- 🔗 Combined prediction engine  

to provide a fast, non-invasive screening solution.

---

## 🚀 Tech Stack

### Frontend
- React.js  
- HTML5  
- CSS3  

### Backend
- Python (Flask)

### Machine Learning
- TensorFlow (Keras)  
- OpenCV  
- Scikit-learn  

### Deployment
- Frontend: Vercel  
- Backend: Render  

---

## ✨ Features

### Spiral Test
- Upload spiral drawing images  
- Processed using OpenCV  
- CNN predicts Parkinson’s probability  

### Voice Test
- Upload `.wav` audio files  
- Extracts:
  - Jitter  
  - Shimmer  
  - HNR  

### Combined Analysis
- Fuses spiral and voice predictions for higher accuracy  

### UI
- Dark/Light mode  
- Responsive design  
