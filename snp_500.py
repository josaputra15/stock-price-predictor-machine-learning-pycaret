

#################
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for plotting
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
lagit(df, 2)
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

x_test = x_test.copy()  
x_test['prediction_LR'] = model.predict(x_test)

x_test['ret'] = df['ret'].loc[x_test.index]


x_test['strat'] = x_test['prediction_LR'] * x_test['ret']

# Plot
(x_test[['strat', 'ret']] + 1).cumprod().plot(figsize=(10, 6))
plt.title("Strategy vs. Market Cumulative Returns")
plt.ylabel("Cumulative Return")
plt.xlabel("Date")
plt.show()

