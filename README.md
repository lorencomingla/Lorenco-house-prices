# 🏡 House Price Prediction

This repository contains a machine learning model that predicts house prices based on input features. The project uses Python, TensorFlow/Keras, and SHAP for explainability. The model is trained on structured housing data and evaluates performance using various metrics.

---

## 📂 Project Structure

```
📁 Lorenco-house-prices/
│── 📜 house_price_model.py          # Main model script
│── 📜 train.csv                      # Training dataset
│── 📜 predictions.csv                # Model predictions
│── 📜 house_price_model.h5           # Saved trained model
│── 📜 preprocessor.pkl               # Preprocessing pipeline
│── 📜 umap_reducer.pkl               # Dimensionality reduction
│── 📜 shap_analysis.png              # SHAP feature importance
│── 📜 README.md                      # Project documentation
```

---

## 🚀 Getting Started

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/lorencomingla/Lorenco-house-prices.git
cd Lorenco-house-prices
```

### **2️⃣ Install Dependencies**
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt` file, manually install key dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap
```

---

## 🏗 Model Training & Evaluation

### **🔹 Training the Model**
Run the following command to train the model:
```bash
python house_price_model.py
```
This script loads data, preprocesses it, trains the model, and saves the results.

### **🔹 Evaluating the Model**
- **Mean Absolute Percentage Error (MAPE):** 10.84%  
- **RMSE (Log Scale):** 0.3276  
- **Feature Importance:** SHAP values are used to interpret model decisions.

---

## 📊 Feature Importance (SHAP)
This project leverages **SHAP (SHapley Additive Explanations)** to understand model predictions.

- The `shap_analysis.png` file provides a **visual explanation of feature importance**.
- SHAP values help **interpret how different features impact predictions**.

---

## 🔄 Model Deployment
- The trained model is saved as `house_price_model.h5`.
- Preprocessing and feature engineering steps are stored in `preprocessor.pkl`.
- The `umap_reducer.pkl` file is used for dimensionality reduction (if applicable).

---

## 🛠 Technologies Used
- **Programming Language:** Python 🐍
- **Libraries:** TensorFlow/Keras, Scikit-Learn, Pandas, SHAP, Matplotlib
- **Machine Learning Concepts:** Regression, Model Explainability, Feature Engineering
- **Version Control:** Git & GitHub  

---

## 📌 Future Improvements
- Improve model performance with hyperparameter tuning.
- Experiment with different regression algorithms (XGBoost, LightGBM).
- Deploy the model using Flask or FastAPI.

---

## 📄 License
This project is licensed under the **MIT License**.

---

## 👤 Author
- **Lorenco Mingla**
- 🌐 [GitHub Profile](https://github.com/lorencomingla)


---

## ⭐ Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

### 🔗 **Project Link**
📌 **GitHub Repo:** [Lorenco-house-prices](https://github.com/lorencomingla/Lorenco-house-prices)

---
