import pandas as pd
from pycaret.regression import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the dataset
data = pd.read_csv('VOO.csv')

# Convert 'Date' column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract features from the Date index
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data['DayOfWeek'] = data.index.dayofweek

# Create the target variable 'NextClose' by shifting the 'Close' prices
data['NextClose'] = data['Close'].shift(-1)

# Remove rows with missing values
data.dropna(inplace=True)

# Split the data into training and testing sets
train = data[data.index < '2022-01-01']  # Training data
test = data[data.index >= '2022-01-01']   # Testing data

# Initialize PyCaret regression setup
reg = setup(data=train, target='NextClose', session_id=123, use_gpu=True)

# Compare different regression models and select the best one
best_model = compare_models()

# Finalize the best model for predictions
finalize_model = finalize_model(best_model)

# Visualize the feature importance of the finalized model
plot_model(finalize_model, plot='feature')

# Generate predictions using the finalized model on the test data
predictions = predict_model(finalize_model, data=test)

def should_buy(current_price, predicted_next_close, threshold=0):
    """
    Determine whether to buy based on the current price and predicted next closing price.

    Parameters:
    - current_price (float): The current price of the stock.
    - predicted_next_close (float): The predicted next closing price of the stock.
    - threshold (float): The minimum profit margin to justify a buy (default is 0).

    Returns:
    - str: "Buy", "Hold", or "Sell" based on the comparison.
    """
    if predicted_next_close > current_price + threshold:
        return "Buy"    # Suggest to buy if the predicted price exceeds the current price plus threshold
    elif predicted_next_close < current_price:
        return "Sell"   # Suggest to sell if the predicted price is lower than the current price
    else:
        return "Hold"   # Suggest to hold if the predicted price is approximately equal to the current price

# Get the last known close price and the latest predicted price
current_price = data['Close'].iloc[-1]              # Current stock price
latest_prediction = predictions['prediction_label'].iloc[-1]  # Latest predicted price

# Determine whether to buy, hold, or sell based on the current price and predicted price
decision = should_buy(current_price, latest_prediction, threshold=0)

# Print the decision along with current and predicted prices
print(f"Current Price: {current_price}, Predicted Next Close: {latest_prediction}, Decision: {decision}")




# # Get the true values of 'NextClose' for the test set
# y_true = test['NextClose']

# # Get the predicted values from the predictions DataFrame
# y_pred = predictions['prediction_label']

# # Calculate Mean Absolute Error (MAE)
# mae = mean_absolute_error(y_true, y_pred)
# print(f"Mean Absolute Error (MAE): {mae}")

# # Calculate Root Mean Squared Error (RMSE)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# print(f"Root Mean Squared Error (RMSE): {rmse}")
