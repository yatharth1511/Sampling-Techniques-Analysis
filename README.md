# Sampling Techniques vs Machine Learning Models

## 📌 Assignment: Sampling Techniques on Imbalanced Data

This project studies the impact of different sampling techniques on the performance of multiple machine learning models using an imbalanced credit card fraud dataset.

The goal is to balance the dataset, generate multiple samples using different sampling strategies, train various ML models, and analyze how sampling affects model accuracy.

---

## 📂 Dataset

- Dataset: Creditcard_data.csv  
- Target variable: `Class`
  - `0` → Normal Transaction  
  - `1` → Fraudulent Transaction  

The dataset is highly imbalanced, which makes it suitable for studying sampling techniques.

---

## 🎯 Objectives

1. Balance the imbalanced dataset  
2. Create five different samples  
3. Apply five different sampling techniques  
4. Train five different machine learning models  
5. Compare accuracy across all combinations  
6. Identify the best sampling–model pair  

---

## 🔁 Sampling Techniques Used

| Sampling ID | Technique | Description |
|------------|----------|-------------|
| Sampling1 | Random Sampling | Random subset selection |
| Sampling2 | Cluster Sampling | Sampling based on KMeans clusters |
| Sampling3 | Bootstrap Sampling | Sampling with replacement |
| Sampling4 | Stratified Sampling | Preserves class distribution |
| Sampling5 | Strategic Sampling | Distance-based informative sampling |

---

## 🤖 Machine Learning Models Used

| Model ID | Algorithm |
|--------|----------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors |
| M5 | Support Vector Machine |

---

## ⚙️ Technologies Used

- Python  
- FastAPI (Backend API)  
- Streamlit (Frontend UI)  
- Scikit-learn  
- Pandas & NumPy  
- Matplotlib  

---

## ▶ Running the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

```

### Step 2: Start backend (integrated with FastAPI)
```bash
cd backend
uvicorn main:app --reload

```

### Step 3: Start frontend (Streamlit)
```bash
cd frontend
streamlit run app.py

```

## Author

### Yatharth Sharma, Roll No- 102303136

