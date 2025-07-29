# Time Series Forecasting Project

## Introduction
This project focuses on time series forecasting using historical Bitcoin prices. The primary goal is to predict future prices using machine learning models. The dataset is sourced from **Coindesk** and includes data from October 1, 2013, to May 18, 2021.

## Dataset
The dataset contains the following columns:
- **Date**: Timestamps of the Bitcoin prices.
- **Closing Price (USD)**: Daily closing price.
- **24h Open (USD)**, **24h High (USD)**, **24h Low (USD)**: Additional features for daily price analysis.

**Total samples**: 2787.

## Workflow
1. **Data Collection**: Downloaded historical data from GitHub.
2. **Data Preprocessing**:
   - Parsed dates using `pandas`.
   - Selected closing prices for analysis.
   - Visualized the data to understand trends.
3. **Train-Test Splitting**: Correctly split the data using an 80-20 ratio, ensuring no data leakage.
4. **Model Development**: Implemented the following models:
   - **Naive Forecast**: Used previous day values as predictions.
   - **Dense Neural Networks**: Horizon and window-based models.
   - **Conv1D Model**: Leveraged convolutional layers for sequential data.
   - **LSTM**: Captured long-term dependencies in time series.
5. **Evaluation**: Metrics used include MAE, MSE, RMSE, MAPE, and MASE.

## Results
- **Naive Forecast**: MAE = 567.98, RMSE = 1071.23.
- **Model 1 (Dense)**: MAE = 568.23, RMSE = 1079.25.
- **Model 2 (Dense, Window 30)**: MAE = 628.52, RMSE = 1129.48.
- **Model 4 (Conv1D)**: MAE = 569.06, RMSE = 1084.39.
- **Model 5 (LSTM)**: Results pending detailed evaluation.

## Code Snippets
### Data Loading and Preprocessing
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
url = "https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"
df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")

# Select closing prices
df['Closing Price (USD)'].plot(figsize=(10, 6), title='Bitcoin Prices', ylabel='Price (USD)')
plt.show()
```

### Naive Forecast
```python
naive_forecast = y_test[:-1]
plt.plot(X_test, y_test, label='Test Data')
plt.plot(X_test[1:], naive_forecast, label='Naive Forecast', linestyle='--')
plt.legend()
plt.show()
```

## Future Work
- Experiment with more advanced architectures like Transformers.
- Implement hyperparameter tuning for optimization.
- Extend the analysis to include seasonality and external factors.

## Acknowledgments
Special thanks to `mrdbourke` for providing the dataset and insights into TensorFlow deep learning.
