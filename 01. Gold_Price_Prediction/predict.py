import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

# Function to load the trained model
def load_trained_model(model_path='trained_model/model.h5'):
    # Load the model
    model = load_model(model_path)
    return model

# Function to prepare the input data for prediction
def prepare_input_data(df, window_size, features, scaler):
    # Take the last window_size rows
    recent_data = df[features].iloc[-window_size:]
    # Scale the data using the new scaler
    scaled_data = scaler.transform(recent_data)
    return scaled_data

# Function to predict future prices
def predict_future_prices(model, recent_data, num_steps, scaler, window_size, features):
    # Ensure recent_data has the correct length
    if len(recent_data) != window_size:
        raise ValueError(f"recent_data must have exactly {window_size} rows.")
    
    # Initialize input sequence for prediction
    input_sequence = recent_data[-window_size:]
    input_sequence = input_sequence.reshape(1, window_size, len(features))

    # Predict future prices
    future_predictions = []
    for _ in range(num_steps):
        # Predict the next price
        next_prediction = model.predict(input_sequence)[0, 0]
        
        # Append the prediction to the results (inverse transform to get actual price)
        future_predictions.append(scaler.inverse_transform([[next_prediction]])[0, 0])
        
        # Update the input sequence: remove the oldest row, add the new prediction
        next_row = input_sequence[0, 1:, :]
        next_prediction_row = np.zeros((1, len(features)))
        next_prediction_row[0, 0] = next_prediction
        input_sequence = np.vstack([next_row, next_prediction_row]).reshape(1, window_size, len(features))
    
    return future_predictions

# Main pipeline function
def gold_price_prediction_pipeline(df, num_days, model_path='model.h5', window_size=60, features=['Price']):
    # Load model
    model = load_trained_model(model_path)

    # Create and fit MinMaxScaler on the entire data for the feature
    scaler = MinMaxScaler()
    scaler.fit(df[features])

    # Prepare the input data (scaled)
    recent_data = prepare_input_data(df, window_size, features, scaler)

    # Predict future prices
    future_prices = predict_future_prices(model, recent_data, num_steps=num_days, scaler=scaler, window_size=window_size, features=features)

    # Generate corresponding dates for the predictions
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

    # Store the results in a dictionary
    future_predictions_dict = {date.date(): round(price, 2) for date, price in zip(future_dates, future_prices)}

    return future_predictions_dict, future_dates, future_prices

# Example usage of the pipeline
num_days_to_predict = 7
model_path = 'model.h5'

# Load dataset (assumes the dataset is always available)
df = pd.read_csv('Gold_Data_Updated.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Call the pipeline function to predict future prices
future_predictions, future_dates, future_prices = gold_price_prediction_pipeline(
    df, num_days=num_days_to_predict, model_path=model_path)

# Display future predictions
print("Future Predicted Prices:")
print(future_predictions)

# Optionally plot the results
plt.figure(figsize=(10, 5))
plt.plot(future_dates, future_prices, marker='o', label="Predicted Prices", color='gold')
plt.title("Future Gold Price Trend", fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.xticks(rotation=45)
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.show()
