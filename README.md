# Sampling Techniques on Imbalanced Credit Card Dataset

## Assignment: Sampling Techniques and Model Evaluation


## Dataset Description

- **Dataset**: Creditcard_data.csv  
- **Target Variable**: `Class`  
  - `0` → Normal transaction  
  - `1` → Fraudulent transaction  

The dataset is highly imbalanced, which makes it suitable for evaluating different sampling strategies.

---

## Objectives

The objectives of this assignment are:

1. To handle class imbalance using different sampling techniques  
2. To generate five sampled datasets  
3. To apply five different machine learning models  
4. To compare model performance using accuracy  
5. To identify the most effective sampling technique for each model  

---

## Methodology

The overall methodology followed in this project is outlined below:

1. **Data Loading**  
   The credit card dataset is loaded and the target variable (`Class`) is separated from the feature set.

2. **Feature Scaling**  
   All features are standardized using `StandardScaler` to ensure fair distance-based computations and model training.

3. **Sampling Techniques**  
   Five different sampling techniques are applied independently to handle the class imbalance:
   - Random Sampling
   - Cluster Sampling
   - Bootstrap Sampling
   - Stratified Sampling
   - Strategic (distance-based) Sampling

4. **Model Training**  
   For each sampled dataset, five machine learning models are trained using a stratified train–test split.

5. **Evaluation**  
   Model performance is evaluated using **accuracy**, and results are compared across all sampling–model combinations.

6. **Analysis**  
   The best sampling technique for each model is identified based on accuracy scores.

---

## Sampling Techniques Used

| Sampling ID | Technique | Description |
|------------|----------|-------------|
| Sampling1 | Random Sampling | Random subset selection |
| Sampling2 | Cluster Sampling | Sampling based on KMeans clustering |
| Sampling3 | Bootstrap Sampling | Sampling with replacement |
| Sampling4 | Stratified Sampling | Maintains class distribution |
| Sampling5 | Strategic Sampling | Distance-based informative sampling |

---

## Machine Learning Models Used

| Model ID | Algorithm |
|--------|----------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | K-Nearest Neighbors |
| M5 | Support Vector Machine |

---

## Implementation Details

- Feature scaling is performed using `StandardScaler`
- Train–test splitting is done using stratification
- Accuracy is used as the evaluation metric
- The complete workflow is contained in a single notebook

---

## Results

- A final accuracy table compares all sampling techniques across models
- The best sampling technique for each model is identified from the results

## Conclusion

This project demonstrates that sampling techniques play a crucial role in handling imbalanced datasets and significantly influence model performance. The results highlight that no single sampling method is universally optimal for all machine learning models.

## Author

### Name: Yatharth Sharma, Roll No: 102303136



