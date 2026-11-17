\#



An end-to-end stock market analytics and price prediction application using Python, Machine Learning, Deep Learning and Streamlit.



\## Project Overview



This project uses historical stock market data from Yahoo Finance to analyze stock trends, compare companies, evaluate machine learning models and estimate the next trading day's stock price.



\## Features



\- Multi-company stock analysis

\- Live retrieval of historical stock data using Yahoo Finance

\- Candlestick charts

\- 20-day and 50-day moving averages

\- Trading volume analysis

\- Company performance comparison

\- Naive Baseline model

\- Linear Regression model

\- Random Forest model

\- LSTM Deep Learning model

\- Model evaluation using MAE, RMSE and R²

\- Actual vs Predicted visualization

\- CSV download functionality



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- TensorFlow / Keras

\- Streamlit

\- Plotly

\- Yahoo Finance (yfinance)



\## Installation



```bash

git clone https://github.com/gundlapallidivya/Google-Stock-Price-Prediction.git

cd Google-Stock-Price-Prediction

pip install -r requirements.txt

```



\## Run the Application



```bash

python -m streamlit run stock\_app.py

```



\## Machine Learning Models



1\. Naive Baseline

2\. Linear Regression

3\. Random Forest

4\. LSTM



The models are evaluated using MAE, RMSE and R². The best model is selected based on validation RMSE.



\## Data Source



Historical stock market data is retrieved from Yahoo Finance using yfinance.



\## Disclaimer



This project is developed for educational purposes. Stock price predictions are estimates and are not guaranteed market prices or financial advice.




