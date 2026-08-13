# 📈 Stock Price Prediction using LSTM

A deep learning project that predicts Apple stock prices using
Long Short-Term Memory (LSTM) networks built with TensorFlow.

## 🚀 Features

- Historical Apple stock analysis
- Data visualization
- LSTM-based prediction
- Actual vs predicted price visualization
- MSE, RMSE and MAE evaluation
- Interactive Streamlit web application

## 🛠️ Tech Stack

Python | TensorFlow | Keras | LSTM | Pandas | NumPy |
Scikit-learn | Matplotlib | Streamlit

## 🧠 Model

The model uses the previous 60 trading days to predict
the next stock closing price.

## 🌐 Live Demo

[Live Demo](https://stock-price-prediction-lstm-pyyqhr7hd9wtstxcg8ycn5.streamlit.app/)

## 📊 Results

The application provides:

- Actual stock prices
- LSTM predicted prices
- MSE
- RMSE
- MAE

## 📂 Project Structure

Stock_Price_Prediction/

- ├── app.py
- ├── main_code.ipynb
- ├── requirements.txt
- ├── all_stocks_5yr.csv
- ├── apple_stock_lstm.keras
- ├── stock_scaler.pkl
- ├── .gitignore
- └── README.md


## 📊 Workflow
Historical Stock Data
        ↓
Data Preprocessing
        ↓
Min-Max Scaling
        ↓
60-Day Sequences
        ↓
LSTM Model
        ↓
Prediction
        ↓
Actual vs Predicted Visualization
