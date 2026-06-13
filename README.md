# 🚗 Car Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=plotly&logoColor=white)](https://matplotlib.org/)

## 📌 Project Overview

The **Car Price Prediction System** is a beginner-friendly Machine Learning web application built with **Streamlit** during the **Oasis Infobyte Data Science Internship**. It predicts the estimated selling price of a used car based on input features such as car age, fuel type, transmission, ownership history, kilometers driven, and present price.

This project demonstrates a complete end-to-end ML workflow: data loading, preprocessing, model training, prediction, visualization, and an interactive UI for users to explore the results.

## ✨ Features

- 🚘 Interactive web interface built with Streamlit
- 📊 Predicts used car selling price in real time
- 🧠 Uses a **Random Forest Regressor** for regression
- 📈 Displays feature importance visualization
- 🔍 Shows actual vs predicted price comparison
- 📁 Includes dataset preview and summary statistics
- 📥 Allows users to download a prediction report as CSV

## 🛠 Technologies Used

- **Python** - core programming language
- **Streamlit** - interactive web app framework
- **Pandas** - data handling and analysis
- **NumPy** - numerical computations
- **Scikit-Learn** - machine learning model and evaluation
- **Matplotlib** - charts and visualizations

## 📂 Dataset Information

The project uses the dataset file **`car data.csv`** included in the repository.

### Dataset Features

- `Car_Name` - name of the car
- `Year` - manufacturing year
- `Selling_Price` - target variable
- `Present_Price` - current market price
- `Driven_kms` - total kilometers driven
- `Fuel_Type` - petrol, diesel, or CNG
- `Selling_type` - dealer or individual
- `Transmission` - manual or automatic
- `Owner` - number of previous owners

### Data Processing

- A new feature, **Car_Age**, is created from the manufacturing year
- Categorical columns are encoded before training
- The target variable is `Selling_Price`

## 🤖 Machine Learning Model

This project uses a **Random Forest Regressor**.

### Why Random Forest Regressor?

- Works well for regression tasks
- Handles non-linear relationships effectively
- Reduces overfitting compared to a single decision tree
- Provides feature importance values for better interpretability

The model is trained directly inside the app using the dataset and then used to generate predictions from user input.

## ⚙ Installation & Run Commands

### 1. Clone or download the project

### 2. Open the project folder

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### Optional: create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn matplotlib
```

## 📁 Project Structure

```bash
Car-Price-Prediction-System/
├── app.py
├── car data.csv
└── README.md
```

## 🚀 Future Enhancements

- Add more datasets for better training performance
- Deploy the app on Streamlit Community Cloud or Hugging Face Spaces
- Improve preprocessing and feature engineering
- Add model comparison with Linear Regression, XGBoost, or Gradient Boosting
- Save the trained model using `pickle` or `joblib`
- Add dark mode and improved UI animations
- Include user authentication and prediction history

## 🧑‍💻 Internship Project Note

This project was developed as part of the **Oasis Infobyte Data Science Internship** to demonstrate practical skills in data preprocessing, supervised machine learning, and interactive application development.

## 📸 Portfolio Ready

This README is designed to look clean and professional for:

- GitHub repositories
- LinkedIn portfolio posts
- Resume project sections
- Internship submissions

---

### Made with ❤️ using Python and Streamlit
