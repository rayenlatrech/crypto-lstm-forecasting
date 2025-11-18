# Crypto LSTM/GRU Forecasting Project

Multivariate Time-Series Modeling • Return Forecasting • Streamlit App

------------------------------------------------------------------------

## 📌 Overview

This project builds a complete **deep learning forecasting pipeline**
for cryptocurrency data using **LSTM** and **GRU** models.\
It demonstrates professional-level Machine Learning workflow steps:

-   Data loading & preprocessing\
-   Feature engineering (log returns, volatility, volume normalization)\
-   Sliding-window dataset creation for sequence models\
-   LSTM & GRU training with validation\
-   Model evaluation\
-   Deployment through a **Streamlit interactive web app**

The project focuses on predicting **next-step log returns**, a realistic
target for financial modeling.

------------------------------------------------------------------------

## 🧠 Key Concepts

### **Log Returns**

We avoid raw price prediction and instead predict log returns:

\[ r_t = `\log`{=tex}`\left`{=tex}(rac{close_t}{close\_{t-1}}ight) \]

This helps with:

-   Stationarity\
-   Avoiding exploding values\
-   Making the target more predictable\
-   Model stability

------------------------------------------------------------------------

## 🛠 Feature Engineering

Each time step includes:

  Feature               Description
  --------------------- -------------------------------------------
  **log_return**        Target and main feature
  **rolling_mean_24**   24-hour rolling mean of returns
  **rolling_std_24**    Volatility indicator
  **volume_norm**       Normalized volume vs 24-hour rolling mean

The model sees these 4 features per time step.

------------------------------------------------------------------------

## 🔍 Model Training

Two architectures were tested:

### ✔ LSTM

### ✔ GRU

Both:

-   Input size = 4 features\
-   Window size = 48 hours\
-   Horizon = 1 (one-step prediction)\
-   MSE loss\
-   Adam optimizer

Models and scalers are saved in:

    /models/
        lstm_btcusdt_best.pt
        gru_btcusdt_best.pt
        scaler_btcusdt.joblib

------------------------------------------------------------------------

## 📊 Evaluation Results

### **Log Return Prediction (LSTM, Multivariate)**

    MAE:  0.00329
    RMSE: 0.00506

This is very similar to a **zero-return baseline**, which means:

> Short-term BTC log returns are extremely hard to predict using only
> OHLCV-derived features.

This conclusion is **realistic** and aligns with academic finance
research.

------------------------------------------------------------------------

## 🌐 Streamlit App

Run:

``` bash
streamlit run app.py
```

The app allows you to:

-   Visualize recent BTC price history\
-   Use the trained LSTM model to predict the *next log return*\
-   Convert it into a next-hour **predicted close price**\
-   Plot this prediction vs historical data

Great for demonstration and portfolio.

------------------------------------------------------------------------

## 📂 Project Structure

    crypto_lstm_project/
    │
    ├── data/                  # raw CSV files (BTCUSDT, other pairs)
    ├── models/                # trained models + scalers
    │
    ├── data_prep.py           # loading & splitting
    ├── dataset.py             # feature engineering + sliding windows
    ├── model.py               # LSTM & GRU definitions
    ├── train.py               # LSTM trainer
    ├── train_gru.py           # GRU trainer (optional)
    ├── evaluate.py            # metrics & plots
    ├── app.py                 # Streamlit app
    │
    └── README.md              # this file

------------------------------------------------------------------------

## 🎯 Main Takeaways

-   Predicting crypto **returns** is extremely challenging\
-   LSTM/GRU models converge to near-zero predictions\
-   This is **normal** and actually validates market efficiency\
-   Feature engineering improves stability but not necessarily
    predictability\
-   The deployed Streamlit app demonstrates full ML pipeline capability

Great for your **GitHub portfolio**, **resume**, and showcasing **real
DS understanding**.

------------------------------------------------------------------------

## 📬 Future Extensions

If you want to extend this project:

-   Forecast **volatility** instead of returns\
-   Add **cross-asset features** from ETH, SOL, etc.\
-   Add **sentiment / funding rate / order-book** features\
-   Extend to **Transformers** or **Temporal Convolutional Networks**\
-   Build a **multi-step forecaster**

------------------------------------------------------------------------

## 👤 Author

**Rayen Latrech**\
2nd Year Data Science Student --- Deep Learning & Time Series Enthusiast

------------------------------------------------------------------------

## ⭐ If you use this project

Please consider giving the repository a star!
