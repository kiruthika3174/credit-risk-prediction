# credit-risk-prediction

# 🏦 Credit Risk Prediction using Machine Learning

This project focuses on building a machine learning model to predict credit risk (high or low) for loan applicants using the German Credit dataset. The final model is deployed through an interactive **Tkinter-based GUI** for real-time risk prediction.

## 📌 Project Overview

- **Dataset**: German Credit Dataset
- **Model Used**: Random Forest Classifier
- **Interface**: Tkinter GUI (desktop app)
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
- **Optimization**: GridSearchCV
- **Explainability**: Feature Importance Visualization

## 🧪 Methodology

1. **Data Preprocessing**:
   - Removed irrelevant columns
   - Encoded categorical variables (one-hot encoding)
   - Target column (`Risk`) encoded to binary (0 = Low, 1 = High)
   - Data split into training and testing sets

2. **Model Building**:
   - Random Forest Classifier trained on processed data
   - Evaluated using classification metrics
   - Saved using `pickle` for future use

3. **Model Optimization**:
   - Used `GridSearchCV` to find the best hyperparameters
   - Best parameters used to retrain the model

4. **Feature Importance**:
   - Feature importance values extracted and visualized

5. **User Interface**:
   - Developed a Tkinter GUI to input applicant details
   - Provides instant prediction result (High or Low Risk)

## 📈 Results

- Achieved **99% accuracy** on test data
- **Classification Report**:
  ```
              precision    recall  f1-score   support

           0       0.97      1.00      0.99        33
           1       1.00      0.99      1.00       167

    accuracy                           0.99       200
    macro avg       0.99      1.00      0.99       200
    weighted avg    1.00      0.99      1.00       200
  ```

## 🖼️ Feature Importance Visualization

The most influential features affecting credit risk prediction were visualized using bar plots, helping improve model explainability.

## 🧠 GridSearchCV Hyperparameter Tuning

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
```

## 🎥 Demo Video

👉 [Click here to watch the demo](https://drive.google.com/file/d/1_TMto5D7kgfpHbzOsn7x4trybg4upmSE/view?usp=drive_link)

## 💻 Technologies Used

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Tkinter (GUI)
- Pickle (model saving)

## ✅ How to Run the Project

1. Clone the repository
2. Install the required libraries from `requirements.txt`
3. Run `main.py` (or GUI file) to launch the Tkinter interface
4. Input data and predict risk level

---

Feel free to ⭐ the repo if you find it useful!
```

---

Let me know if you want me to generate a `requirements.txt` file or customize this README for a **Streamlit** version later.
