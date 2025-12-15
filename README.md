# 📈 Solana Price Forecasting (LSTM vs. Linear Regression)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Project Overview
This project aims to predict the daily price of **Solana (SOL)** cryptocurrency using *Machine Learning* and *Deep Learning* approaches. Given the high volatility of the crypto market, this project compares the performance of a baseline model (**Linear Regression**) with a *Deep Learning* model (**Long Short-Term Memory / LSTM**) to evaluate their effectiveness in capturing *time-series* patterns.

This project was developed as part of the **DBS Coding Camp 2024: Expert Class**.

## 📊 Methodology
The project workflow includes:
1.  **Data Understanding**: Fetching historical data from Investing.com (July 2020 - Nov 2024).
2.  **Exploratory Data Analysis (EDA)**:
    * Univariate & Multivariate Analysis.
    * Time Series Analysis (Decomposition, ACF/PACF).
    * Technical Indicators (MA, EMA, MACD, Bollinger Bands).
3.  **Data Preprocessing**: Handling missing values, data normalization (MinMax Scaler), and sequence generation for LSTM.
4.  **Modeling**:
    * Baseline Model: Linear Regression.
    * Main Model: LSTM (Sequential with Dropout layers).
5.  **Evaluation**: Using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) metrics.

## 🛠️ Technologies Used
* **Language**: Python
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn, Plotly
* **Machine Learning**: Scikit-Learn
* **Deep Learning**: TensorFlow (Keras)
* **Statistics**: Statsmodels

## 📈 Evaluation Results
Based on testing on the test data, the LSTM model demonstrated significantly superior performance compared to Linear Regression in capturing price trends.

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) |
| :--- | :--- | :--- |
| **Linear Regression** | 96.34 | 100.02 |
| **LSTM (Deep Learning)** | **4.99** | **6.46** |

> **Conclusion:** LSTM successfully reduced prediction error significantly due to its ability to remember long-term patterns (*long-term dependencies*) which cannot be captured by simple linear regression.

## 🚀 How to Run the Project

### Prerequisites
Ensure you have Python installed. It is recommended to use a *virtual environment*.

### Installation
1.  Clone this repository:
    ```bash
    git clone [https://github.com/username/solana-price-forecasting.git](https://github.com/username/solana-price-forecasting.git)
    cd solana-price-forecasting
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
You can run the Jupyter Notebook or the Python script:

* **Jupyter Notebook:**
    ```bash
    jupyter notebook notebooks/solana_forecasting.ipynb
    ```
* **Python Script:**
    ```bash
    python notebooks/solana_forecasting.py
    ```

## 📂 Folder Structure
```text
├── data/               # CSV Dataset (solana.csv)
├── notebooks/          # Source code (ipynb & py)
├── reports/            # Full analysis report (Markdown)
├── requirements.txt    # List of libraries
└── README.md           # Project documentation
