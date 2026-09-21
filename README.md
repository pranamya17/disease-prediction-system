# disease-prediction-system
disease prediction system using Machine Learning in Healthcare Analytics 


# Student Performance Prediction System using Machine Learning

## 📌 Project Description

The Student Performance Prediction System is a machine learning project that predicts whether a student is likely to **Pass or Fail** based on academic factors such as study hours, attendance, previous marks, assignment scores, and internal marks.

The system also identifies students who may be **at risk of failing**, helping teachers identify students who may need additional support.

## 🎯 Objectives

- Predict student performance as Pass or Fail.
- Identify students who are at risk of failing.
- Analyze factors related to student performance.
- Compare different machine learning models.
- Evaluate model performance using standard evaluation metrics.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Excel

## 📊 Dataset

A sample dataset containing **100 student records** was created for this project.

The dataset contains:

- Student ID
- Study Hours
- Attendance
- Previous Marks
- Assignments
- Internal Marks
- Final Result

> **Note:** The dataset used in this project is a synthetic/sample dataset created for demonstration purposes.

## 🔄 Project Workflow

1. Data Collection
2. Data Preprocessing
3. Feature Selection
4. Data Normalization
5. Model Training
6. Model Testing
7. Student Performance Prediction
8. At-Risk Student Identification
9. Model Evaluation
10. Result Visualization

## 🤖 Machine Learning Models

### 1. Logistic Regression
Used to predict whether a student will Pass or Fail.

### 2. Decision Tree
Used to classify students based on their academic features.

### 3. Normalized Logistic Regression
Logistic Regression was also trained using normalized numerical features.

## 📈 Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 90% |
| Decision Tree | 75% |
| Normalized Logistic Regression | 90% |

### Logistic Regression Evaluation

- Accuracy: 90%
- Precision: 90%
- Recall: 90%
- F1-Score: 90%

### Decision Tree Evaluation

- Accuracy: 75%
- Precision: 69.23%
- Recall: 90%
- F1-Score: 78.26%

## ⚠️ At-Risk Students

The system predicted:

- Total Students: **100**
- Students At Risk: **51**
- Students Not At Risk: **49**

## 📊 Visualizations

The project includes:

- Study Hours vs Previous Marks
- Attendance vs Previous Marks
- Correlation Heatmap
- Predicted Student Performance
- Confusion Matrix
- Model Accuracy Comparison

## 📁 Project Files

```text
Student-Performance-Prediction-System/
│
├── student_performance_prediction.ipynb
├── student_performance_results.xlsx
└── README.md
