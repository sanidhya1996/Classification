# Airline Passenger Referral Prediction 🚀✈️  

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![ML Framework](https://img.shields.io/badge/Machine%20Learning-Project-success.svg)](https://scikit-learn.org/)  
[![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen.svg)]()  

## 📌 Project Overview  

This project predicts whether a passenger will **recommend an airline** to others, based on various review parameters like cabin service, seat comfort, and value-for-money. By analyzing the dataset, we aim to extract insights and build machine learning models to provide accurate predictions.  

---

## 📋 Problem Statement  

The dataset contains airline reviews spanning from **2006 to 2019**, with both **numerical ratings** and **text-based feedback**. The primary objective is:  

👉 **"Predict whether passengers will recommend the airline based on their ratings of core experience factors (seat comfort, food/beverage, cabin service, etc.) and demographics."**

---

## 📂 Dataset Overview  

The dataset includes **131,895 reviews** with the following features:  

### Key Columns:  
- `airline`: Airline name.  
- `overall`: Overall rating (1–10).  
- `traveller_type`: Type of traveler (e.g., Solo, Couple).  
- `cabin`: Type of flight cabin (Economy, Business, etc.).  
- `value_for_money`, `seat_comfort`, `cabin_service`, etc.: Ratings for specific airline services.  
- `recommended`: Target variable, whether or not a passenger recommends the airline (Yes/No).  

### Challenges:  
1. **Missing Values** in key columns like `traveller_type`, `cabin`, and `value_for_money`.  
2. **Imbalanced Data** – Disproportionate classes in the `recommended` column.  

---

## 🛠️ Workflow  

The project follows the **CRISP-DM methodology**:  

### 1. **Data Preprocessing**  
   - Missing value imputation using **KNN Imputer** for numerical columns and **forward fill** for categorical columns.  
   - Outliers treated using **z-score techniques** for numerical ratings.  
   - **Dropped irrelevant features** (e.g., `review_date`, `airline`, free-text reviews).  

### 2. **Exploratory Data Analysis (EDA)**  
   - Data exploration to identify correlations, trends, and target relationships:  
     - Visualized data using **plots like histograms, bar charts, pair plots**, and a **correlation heatmap**.  
     - Found that `value_for_money` and `cabin_service` ranked highest among features correlated with recommendations.  

### 3. **Feature Engineering**  
   - Removed multicollinear features using **Variance Inflation Factor (VIF)**.  
   - Applied categorical encoding (e.g., One-Hot Encoding for `cabin`, `traveller_type`).  

### 4. **Model Development**  
   - Used several classification algorithms:  
     1. Logistic Regression  
     2. Decision Tree  
     3. Random Forest  
     4. K-Nearest Neighbors (KNN)  
   - **Hyperparameter Tuning** using `GridSearchCV` to identify best-performing models.  

### 5. **Model Evaluation**  
   - Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.  

---

## 📊 Exploratory Data Analysis Highlights  

1. **Positive Recommendations:** Passengers who rate high for `value_for_money` and `cabin_service` are more likely to recommend the airline.  
2. **Traveler Type:** Passengers traveling Solo or for leisure purposes have the most reviews and recommendations.  
3. **Cabin Type:** Economy class passengers reviewed the most, though Business class received the highest average ratings.  

---

## ⚙️ Machine Learning Models  

### **Model Comparison**  

| Model                         | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |  
|-------------------------------|----------|-----------|---------|----------|---------|  
| Logistic Regression           | 95.21%   | 94.57%    | 92.14%  | 93.84%   | 95.47%  |  
| Decision Tree                 | 93.38%   | 92.12%    | 91.73%  | 91.92%   | 94.22%  |  
| Random Forest                 | **95.32%**| **95.33%** | **94.83%** | **95.07%** | **95.79%** |  
| K-Nearest Neighbors (KNN)     | 95.07%   | 94.81%    | 93.42%  | 94.11%   | 95.06%  |  

**Selected Model:**  
- **Random Forest**: Combines high accuracy with well-balanced precision, recall, and overall robustness.  

### Feature Importance Analysis  
Using **SHAP (SHapley Additive ExPlanations):**  
1. **Value_for_money** was the most influential feature in recommendations.  
2. **Cabin service** and **seat comfort** followed as next significant contributors.  

---

## 📌 Results and Insights  

- The **Random Forest model** achieved the best overall performance with an accuracy of 95.32%.  
- **Key actionable insights** for improving recommendations:  
  - Improving `value_for_money` and `cabin_service` can directly boost airline referrals.  
  - Business strategy should focus more on the **Economy class** segment.  

---

## 🔮 Future Enhancements  

1. **Sentiment Analysis** on `customer_review` text column to extract additional insights about passenger feedback.  
2. Incorporate **time-series analysis** to observe yearly or seasonal trends in recommendations.  
3. Experiment with **deep learning** models like Neural Networks or **advanced ensemble methods** like XGBoost.  

---

## 🛠️ Setup  

Follow the steps below to set up and run the project on your local machine:  

### Prerequisites  
Make sure you have the following installed:  
- Python 3.7 or above  
- Packages: numpy, pandas, matplotlib, seaborn, scikit-learn, shap  

To install the required libraries, use:  
```bash
pip install -r requirements.txt


## 📬 Contact

For any questions, suggestions, or collaboration opportunities, feel free to reach out:

- **Email**: [sanidhyashekhar1996@gmail.com](mailto:sanidhyashekhar1996@gmail.com)

---

## 🌟 Acknowledgments

- Dataset scraped in Spring 2019 for airline reviews between 2006 and 2019.  
- Special thanks to the open-source libraries used, including **Pandas**, **Seaborn**, and **Scikit-learn**, for making data analysis and machine learning easier.  
- Inspiration drawn from similar data science projects and public documentation.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
