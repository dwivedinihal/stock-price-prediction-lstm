import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

model = tf.keras.models.load_model(
    "apple_stock_lstm.keras"
)

scaler = joblib.load(
    "stock_scaler.pkl"
)

data = pd.read_csv(
    "all_stocks_5yr.csv"
)

data["date"] = pd.to_datetime(
    data["date"]
)

apple = data[
    data["Name"] == "AAPL"
].copy()

st.set_page_config(
    page_title="Apple Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Apple Stock Price Prediction")
st.markdown(
    "### TensorFlow + LSTM Based Stock Price Prediction"
)

st.divider()

st.subheader("📊 Apple Historical Stock Price")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    apple["date"],
    apple["close"]
)

ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.set_title("Apple Stock Closing Price")

st.pyplot(fig)

close_data = apple[["close"]]

dataset = close_data.values

training_size = int(
    len(dataset) * 0.80
)

scaled_data = scaler.transform(dataset)

test_data = scaled_data[
    training_size - 60:
]

x_test = []

y_test = dataset[
    training_size:
]

for i in range(60, len(test_data)):

    x_test.append(
        test_data[i-60:i, 0]
    )

x_test = np.array(x_test)

x_test = np.reshape(
    x_test,
    (x_test.shape[0], x_test.shape[1], 1)
)

predictions = model.predict(
    x_test,
    verbose=0
)

predictions = scaler.inverse_transform(
    predictions
)

test_dates = apple[
    "date"
].iloc[training_size:].values

results = pd.DataFrame({
    "Date": test_dates,
    "Actual": y_test.flatten(),
    "Predicted": predictions.flatten()
})

st.subheader(
    "📈 Actual vs Predicted Stock Price"
)

fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    results["Date"],
    results["Actual"],
    label="Actual Price"
)

ax.plot(
    results["Date"],
    results["Predicted"],
    label="Predicted Price"
)

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")

ax.legend()

plt.xticks(rotation=45)

st.pyplot(fig)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    predictions
)

st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MSE",
    f"{mse:.2f}"
)

col2.metric(
    "RMSE",
    f"{rmse:.2f}"
)

col3.metric(
    "MAE",
    f"{mae:.2f}"
)

st.divider()

st.subheader("🧠 About the Model")

st.write("""
This project uses a Long Short-Term Memory (LSTM) neural network
to predict Apple stock closing prices.

The model uses the previous 60 trading days as input
to predict the next stock price.

Technology Stack:
- Python
- TensorFlow
- Keras
- LSTM
- Pandas
- NumPy
- Scikit-learn
- Streamlit
""")