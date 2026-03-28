# 🏥 Early Disease Prediction System
### AI-Powered Diabetes Risk Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](https://github.com)
[![Hackathon](https://img.shields.io/badge/Project_Jupyter-Hackathon-028090?style=for-the-badge)](https://unstop.com)

> A full-stack AI/ML web application that predicts diabetes risk from basic health parameters — making early detection accessible to everyone.

---

## 📸 App Preview

> Run the app locally (see below) to see the full interactive UI with sliders, risk scores, charts, and more.

---

## 📁 Project Structure

```
disease_predictor/
│
├── app.py                  ← Main Streamlit application (all ML + UI code)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 📋 Dataset

| Property       | Details                                                                 |
|----------------|-------------------------------------------------------------------------|
| **Name**       | Pima Indians Diabetes Database                                          |
| **Source**     | [Kaggle — UCI Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Records**    | 768 patient records                                                     |
| **Features**   | 8 clinical health parameters                                            |
| **Target**     | Outcome (1 = Diabetic, 0 = Not Diabetic)                                |
| **Origin**     | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)|

### Features Used

| Feature                    | Description                                     | Unit       |
|----------------------------|-------------------------------------------------|------------|
| Pregnancies                | Number of times pregnant                        | count      |
| Glucose                    | Plasma glucose concentration (2-hour OGTT)      | mg/dL      |
| BloodPressure              | Diastolic blood pressure                        | mm Hg      |
| SkinThickness              | Triceps skinfold thickness                      | mm         |
| Insulin                    | 2-hour serum insulin                            | μU/ml      |
| BMI                        | Body Mass Index (weight/height²)                | kg/m²      |
| DiabetesPedigreeFunction   | Genetic risk score based on family history      | score      |
| Age                        | Patient age                                     | years      |

---

## 🔑 Key Findings

- 🩸 **Glucose** is the single strongest predictor of diabetes risk
- ⚖️ **BMI** and **Age** are the next most important factors
- 🧬 **Diabetes Pedigree Function** captures family history risk effectively
- 📊 **~35%** of records in the dataset are diabetic cases
- 🤖 **Random Forest (200 trees)** outperforms single decision trees significantly

---

## 🧠 ML Pipeline

| Stage | Task | Tool |
|-------|------|------|
| 1 | Data loading and exploration | Pandas, NumPy |
| 2 | Feature normalization | StandardScaler (Scikit-learn) |
| 3 | Train/test split (80/20) | train_test_split |
| 4 | Model training | RandomForestClassifier |
| 5 | Evaluation (accuracy + feature importance) | Scikit-learn |
| 6 | Interactive web UI | Streamlit |
| 7 | Data visualizations | Matplotlib |

---

## 📊 App Features

| Feature | Description |
|---------|-------------|
| 🎯 Risk Prediction | Real-time diabetes probability score with color-coded result |
| 📋 Input Summary | Your values vs normal clinical ranges |
| 📊 Feature Importance | Horizontal bar chart showing top predictors |
| 📉 Glucose Distribution | Histogram comparing diabetic vs non-diabetic glucose levels |
| 🔗 Correlation Heatmap | Full feature correlation matrix |
| ℹ️ About Tab | Tech stack, dataset info, and Kaggle link |

---

## 🚀 How to Run

### Step 1 — Clone or download this folder

```bash
git clone https://github.com/YOUR_USERNAME/disease-predictor.git
cd disease-predictor
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Use the real dataset

Download `diabetes.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Place it in the project folder, then in `app.py` replace the synthetic data block with:
```python
df = pd.read_csv("diabetes.csv")
```

### Step 4 — Launch the app

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501** 🎉

---

## 📦 Requirements

```
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
matplotlib==3.9.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Scikit-learn | ML model (Random Forest) + preprocessing |
| Pandas & NumPy | Data handling and manipulation |
| Matplotlib | Data visualization charts |
| Streamlit | Interactive web application framework |

---

## 💡 Business / Social Impact

1. **Early detection saves lives** — Diabetes diagnosed early is far more manageable
2. **Rural accessibility** — Works on any browser, no installation for end users
3. **Cost reduction** — Reduces burden of expensive late-stage treatment
4. **Scalable** — Model can be extended to predict heart disease, kidney disease, and more
5. **Awareness** — Educates users about which health parameters matter most

---

## 🏆 Event Details

| Field | Info |
|-------|------|
| **Event** | Project Jupyter Hackathon |
| **Organizer** | Army Institute of Technology |
| **Domain** | Healthcare — Artificial Intelligence & Machine Learning |
| **Year** | 2026 |
| **Platform** | [Unstop](https://unstop.com) |

---

## 📄 Disclaimer

This application is developed for the **Project Jupyter Hackathon** and is strictly for **educational and demonstration purposes**. It is **NOT a certified medical device** and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 👤 Author

**[Your Name]**
- 🏫 Army Institute of Technology
- 💻 GitHub: [Meghana Uppar](https://github.com/Meghana-b-u)
- 🔗 LinkedIn: [Meghana Uppar ](https://www.linkedin.com/in/meghana-uppar-374603267)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

⭐ **Star this repo if you found it helpful!**
