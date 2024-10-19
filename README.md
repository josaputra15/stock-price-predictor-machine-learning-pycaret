# Stock Price Predictor

This project aims to predict the next closing price of a stock using historical data. It utilizes the PyCaret library for machine learning and regression analysis. The model helps investors make informed decisions about buying, holding, or selling stocks based on predicted price movements.

## Dataset

The dataset used in this project is `VOO.csv`, which contains historical stock prices for the Vanguard S&P 500 ETF (VOO). The dataset includes the following columns:

- **Date**: The date of the stock price.
- **Open**: The price at which the stock opened on a given day.
- **High**: The highest price of the stock during the day.
- **Low**: The lowest price of the stock during the day.
- **Close**: The price at which the stock closed on a given day.
- **Volume**: The number of shares traded on that day.

Make sure to place the `VOO.csv` file in the same directory as the script before running the code.

## Requirements

To run this project, ensure you have the following Python packages installed:

- pandas
- pycaret
- matplotlib (for visualizations)

You can install the required packages using pip:

```bash
pip install pandas pycaret matplotlib
```

## Usage

1. **Load the Dataset**: The script reads the `VOO.csv` file into a Pandas DataFrame.
2. **Preprocess the Data**: It processes the date, extracts relevant features, and creates a target variable for prediction.
3. **Model Training**: The script initializes the PyCaret regression setup and compares different regression models to find the best one.
4. **Prediction**: The best model is finalized and used to predict the next closing prices.
5. **Decision Making**: The `should_buy` function evaluates whether to buy, sell, or hold based on the current price and predicted next closing price.

## Functionality

### should_buy

This function determines whether to buy, hold, or sell based on the current stock price and the predicted next closing price.

**Parameters:**

- `current_price` (float): The current price of the stock.
- `predicted_next_close` (float): The predicted next closing price of the stock.
- `threshold` (float): The minimum profit margin to justify a buy (default is 0).

**Returns:**

- "Buy" if the predicted price exceeds the current price plus the threshold.
- "Sell" if the predicted price is lower than the current price.
- "Hold" if the predicted price is approximately equal to the current price.

## Example Output

When you run the script, it will output the current price, predicted next closing price, and the decision to buy, hold, or sell:

```
Current Price: [current_price], Predicted Next Close: [predicted_next_close], Decision: [decision]
```
