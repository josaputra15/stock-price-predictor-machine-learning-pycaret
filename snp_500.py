import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download data
df = yf.download('^GSPC', start='2010-01-01')

# Calculate daily returns
df['ret'] = df.Close.pct_change()

# Define function to create lagged returns
def lagit(df, lags):
    for i in range(1, lags + 1):
        df['Lag_' + str(i)] = df['ret'].shift(i)
    return ['Lag_' + str(i) for i in range(1, lags + 1)]

# Create lagged features and the target variable
lagit(df, 10)
df['direction'] = np.where(df.ret > 0, 1, 0)
features = lagit(df, 3)
df.dropna(inplace=True)

# Define X and y for logistic regression
x = df[features]
y = df['direction']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# Fit logistic regression model
model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

# Make predictions on the test set
x_test = x_test.copy()
x_test['prediction_LR'] = model.predict(x_test)
x_test['ret'] = df['ret'].loc[x_test.index]
x_test['direction'] = df['direction'].loc[x_test.index]  # Add the 'direction' column from original df
x_test['strat'] = x_test['prediction_LR'] * x_test['ret']

# Function to calculate and print model accuracy
def calculate_accuracy(data):
    """
    Calculate and print the accuracy of the model's predictions compared to the actual returns.

    Parameters:
    - data: DataFrame with columns 'direction' (actual) and 'prediction_LR' (predicted).
    """
    correct_predictions = (data['prediction_LR'] == data['direction']).sum()
    total_predictions = len(data)
    accuracy = correct_predictions / total_predictions * 100
    print(f"Model Accuracy: {accuracy:.2f}%")


# calculate_accuracy(x_test)

# Function to plot strategy up to a specified end date
def plot_strategy_until(data, end_date):
    """
    Plot cumulative returns of the strategy until a specified end date.

    Parameters:
    - data: DataFrame with the 'strat' and 'ret' columns for cumulative returns calculation.
    - end_date: str, Date in 'YYYY-MM-DD' format to limit the plot up to this date.
    """
    # Filter data up to the end date
    data_until = data[data.index <= end_date]

    # Plot cumulative returns
    (data_until[['strat', 'ret']] + 1).cumprod().plot(figsize=(10, 6))
    plt.title(f"Strategy vs. Market Cumulative Returns (up to {end_date})")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.show()


plot_strategy_until(x_test, '2024-12-31')

# Function to predict future market direction
def predict_future_direction(model, df, features, days=5):
    """
    Predicts whether the market will go up or down for the specified number of future days.

    Parameters:
    - model: Trained logistic regression model.
    - df: DataFrame containing historical stock data.
    - features: List of feature columns for the model.
    - days: Number of future days to predict.

    Prints:
    - Predicted market direction for each future day.
    """
    
    last_row = df[features].iloc[-1]
    
    for day in range(1, days + 1):
        future_features = last_row.values.reshape(1, -1)
        prediction = model.predict(future_features)[0]
        
        direction = "Up" if prediction == 1 else "Down"
        print(f"Day {day}: {direction}")


#predict_future_direction(model, df, features, days=5)