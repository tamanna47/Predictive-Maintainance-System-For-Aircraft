# ✈️ Predictive Maintenance System for Aircraft  
An advanced ensemble deep learning–based system designed to predict **aircraft engine health**, detect early-stage failures, and support data-driven maintenance decision-making.
## 📌 Overview
This project implements an **ensemble of CNN, LSTM, and Random Forest models** to analyze sensor data and predict the health status of aircraft engines.  
The system is optimized for **accuracy, stability, and real-time maintenance applications**
## 🎯 Objectives
- Predict aircraft engine condition with **95%+ accuracy**  
- Detect anomalies before actual failure  
- Reduce downtime and maintenance costs  
- Enable predictive, not reactive, maintenance  
## ✨ Features
- 🧠 **Ensemble Model:** CNN + LSTM + Random Forest  
- 📊 **Real-time health score & prediction**  
- 🔍 **Sensor-driven insights** (temp, pressure, vibration, RPM, fuel flow, etc.)  
- ⚙️ **Modular code for training, testing, and deployment**  
- 📈 **Graphs: accuracy, loss curves, confusion matrix**
## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Scikit-Learn**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
## 🏗️ System Architecture
Sensor Data → Preprocessing → CNN + LSTM + Random Forest → Ensemble Layer → Engine Health Prediction
## 📂 Project Structure
Predictive-Maintainance-System-For-Aircraft/
│── src/
│── models/
│── data/
│── results/
│── main.py
│── predict.py
│── README.md

---## 🚀 Installation
### Clone the repository
git clone https://github.com/yourusername/Predictive-Maintainance-System-For-Aircraft.git
cd Predictive-Maintainance-System-For-Aircraft
**Install dependencies**
python main.py
**Run prediction**
python predict.py --rpm 2400 --temp 650 --vibration 0.03 --pressure 2.5

