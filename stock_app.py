
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #58a6ff;
}

div[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 15px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"] {
    color: #c9d1d9;
}

div[data-testid="stMetricValue"] {
    color: #58a6ff;
}

.dashboard-subtitle {
    color: #aab7c4;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# CONSTANTS
# =========================================================

LOOKBACK = 60

COMPANIES = {
    "Google (Alphabet)": "GOOGL",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Meta": "META",
    "Netflix": "NFLX",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Wipro": "WIPRO.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "ITC": "ITC.NS",
    "State Bank of India": "SBIN.NS",
    "Adani Enterprises": "ADANIENT.NS"
}


# =========================================================
# CURRENCY FUNCTIONS
# =========================================================

@st.cache_data(ttl=3600)
def get_usd_to_inr_rate():

    try:
        data = yf.Ticker("USDINR=X").history(
            period="5d",
            auto_adjust=True
        )

        if not data.empty:
            return float(data["Close"].dropna().iloc[-1])

    except Exception:
        pass

    return 85.0


def convert_to_inr(value, ticker):

    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return float(value)

    return float(value) * get_usd_to_inr_rate()


def currency_label(ticker):

    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "INR"

    return "INR (converted from USD)"


# =========================================================
# DOWNLOAD STOCK DATA
# =========================================================

@st.cache_data(ttl=3600)
def download_stock_data(ticker):

    data = yf.Ticker(ticker).history(
        period="5y",
        interval="1d",
        auto_adjust=True
    )

    if data.empty:
        raise ValueError(
            f"No historical data found for {ticker}"
        )

    required_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    data = data[required_columns].copy()
    data.dropna(inplace=True)

    if len(data) <= LOOKBACK + 20:
        raise ValueError("Not enough historical data.")

    return data


# =========================================================
# CREATE SEQUENCES
# =========================================================

def make_sequences(values):

    X = []
    y = []

    for i in range(LOOKBACK, len(values)):

        X.append(values[i - LOOKBACK:i, 0])
        y.append(values[i, 0])

    return np.array(X), np.array(y)


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


# =========================================================
# BUILD LSTM
# =========================================================

def build_lstm_model():

    model = Sequential([
        LSTM(
            50,
            return_sequences=True,
            input_shape=(LOOKBACK, 1)
        ),
        Dropout(0.2),

        LSTM(
            50,
            return_sequences=False
        ),
        Dropout(0.2),

        Dense(25),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# =========================================================
# TRAIN AND EVALUATE MODELS
# =========================================================

@st.cache_resource(show_spinner=False)
def train_and_evaluate_models(ticker):

    data = download_stock_data(ticker)

    prices = data[["Close"]].values.astype(float)

    train_end = int(len(prices) * 0.70)
    val_end = int(len(prices) * 0.85)

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    scaler.fit(prices[:train_end])

    scaled_data = scaler.transform(prices)

    X_all, y_all = make_sequences(scaled_data)

    target_indices = np.arange(
        LOOKBACK, len(prices)
    )

    train_mask = target_indices < train_end

    val_mask = (
        (target_indices >= train_end) &
        (target_indices < val_end)
    )

    test_mask = target_indices >= val_end

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    X_val = X_all[val_mask]
    y_val = y_all[val_mask]

    X_test = X_all[test_mask]
    y_test = y_all[test_mask]

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError(
            "Train, validation or test dataset is empty."
        )

    y_val_actual = scaler.inverse_transform(
        y_val.reshape(-1, 1)
    ).ravel()

    y_test_actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    results = {}
    validation_predictions = {}
    test_predictions = {}
    trained_models = {}

    # -----------------------------------------------------
    # 1. NAIVE BASELINE
    # -----------------------------------------------------

    naive_val_scaled = X_val[:, -1]
    naive_test_scaled = X_test[:, -1]

    naive_val = scaler.inverse_transform(
        naive_val_scaled.reshape(-1, 1)
    ).ravel()

    naive_test = scaler.inverse_transform(
        naive_test_scaled.reshape(-1, 1)
    ).ravel()

    results["Naive Baseline"] = {
        "Validation": calculate_metrics(
            y_val_actual, naive_val
        ),
        "Test": calculate_metrics(
            y_test_actual, naive_test
        )
    }

    validation_predictions["Naive Baseline"] = naive_val
    test_predictions["Naive Baseline"] = naive_test

    # -----------------------------------------------------
    # PREPARE SKLEARN INPUTS
    # -----------------------------------------------------

    X_train_flat = X_train.reshape(
        len(X_train), LOOKBACK
    )

    X_val_flat = X_val.reshape(
        len(X_val), LOOKBACK
    )

    X_test_flat = X_test.reshape(
        len(X_test), LOOKBACK
    )

    # -----------------------------------------------------
    # 2. LINEAR REGRESSION
    # -----------------------------------------------------

    linear_model = LinearRegression()

    linear_model.fit(
        X_train_flat,
        y_train
    )

    linear_val_scaled = linear_model.predict(
        X_val_flat
    )

    linear_test_scaled = linear_model.predict(
        X_test_flat
    )

    linear_val = scaler.inverse_transform(
        linear_val_scaled.reshape(-1, 1)
    ).ravel()

    linear_test = scaler.inverse_transform(
        linear_test_scaled.reshape(-1, 1)
    ).ravel()

    results["Linear Regression"] = {
        "Validation": calculate_metrics(
            y_val_actual, linear_val
        ),
        "Test": calculate_metrics(
            y_test_actual, linear_test
        )
    }

    validation_predictions["Linear Regression"] = linear_val
    test_predictions["Linear Regression"] = linear_test

    trained_models["Linear Regression"] = linear_model

    # -----------------------------------------------------
    # 3. RANDOM FOREST
    # -----------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train_flat,
        y_train
    )

    rf_val_scaled = rf_model.predict(
        X_val_flat
    )

    rf_test_scaled = rf_model.predict(
        X_test_flat
    )

    rf_val = scaler.inverse_transform(
        rf_val_scaled.reshape(-1, 1)
    ).ravel()

    rf_test = scaler.inverse_transform(
        rf_test_scaled.reshape(-1, 1)
    ).ravel()

    results["Random Forest"] = {
        "Validation": calculate_metrics(
            y_val_actual, rf_val
        ),
        "Test": calculate_metrics(
            y_test_actual, rf_test
        )
    }

    validation_predictions["Random Forest"] = rf_val
    test_predictions["Random Forest"] = rf_test

    trained_models["Random Forest"] = rf_model

    # -----------------------------------------------------
    # 4. LSTM
    # -----------------------------------------------------

    X_train_lstm = X_train.reshape(
        -1, LOOKBACK, 1
    )

    X_val_lstm = X_val.reshape(
        -1, LOOKBACK, 1
    )

    X_test_lstm = X_test.reshape(
        -1, LOOKBACK, 1
    )

    lstm_model = build_lstm_model()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    lstm_model.fit(
        X_train_lstm,
        y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    lstm_val_scaled = lstm_model.predict(
        X_val_lstm,
        verbose=0
    ).ravel()

    lstm_test_scaled = lstm_model.predict(
        X_test_lstm,
        verbose=0
    ).ravel()

    lstm_val = scaler.inverse_transform(
        lstm_val_scaled.reshape(-1, 1)
    ).ravel()

    lstm_test = scaler.inverse_transform(
        lstm_test_scaled.reshape(-1, 1)
    ).ravel()

    results["LSTM"] = {
        "Validation": calculate_metrics(
            y_val_actual, lstm_val
        ),
        "Test": calculate_metrics(
            y_test_actual, lstm_test
        )
    }

    validation_predictions["LSTM"] = lstm_val
    test_predictions["LSTM"] = lstm_test

    trained_models["LSTM"] = lstm_model

    # -----------------------------------------------------
    # SELECT BEST MODEL USING VALIDATION RMSE
    # -----------------------------------------------------

    best_model_name = min(
        results,
        key=lambda name: results[name]["Validation"][1]
    )

    # -----------------------------------------------------
    # NEXT TRADING DAY ESTIMATE
    # -----------------------------------------------------

    latest_sequence = scaled_data[-LOOKBACK:].reshape(
        1, LOOKBACK
    )

    if best_model_name == "Naive Baseline":

        next_scaled = latest_sequence[0, -1]

    elif best_model_name == "LSTM":

        latest_lstm = latest_sequence.reshape(
            1, LOOKBACK, 1
        )

        next_scaled = trained_models[
            "LSTM"
        ].predict(
            latest_lstm,
            verbose=0
        )[0, 0]

    else:

        next_scaled = trained_models[
            best_model_name
        ].predict(
            latest_sequence
        )[0]

    next_prediction = scaler.inverse_transform(
        np.array([[next_scaled]])
    )[0, 0]

    return {
        "data": data,
        "results": results,
        "best_model": best_model_name,
        "next_prediction": float(next_prediction),
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "y_val_actual": y_val_actual,
        "y_test_actual": y_test_actual,
        "val_dates": data.index[target_indices[val_mask]],
        "test_dates": data.index[target_indices[test_mask]]
    }


# =========================================================
# COMPANY COMPARISON DATA
# =========================================================

@st.cache_data(ttl=3600)
def download_comparison_data(tickers):

    comparison = pd.DataFrame()

    for ticker in tickers:

        try:
            stock_data = yf.Ticker(ticker).history(
                period="1y",
                interval="1d",
                auto_adjust=True
            )

            if not stock_data.empty:

                close = stock_data["Close"].dropna()

                if len(close) > 0:

                    normalized = (
                        close / close.iloc[0]
                    ) * 100

                    comparison[ticker] = normalized

        except Exception:
            continue

    return comparison


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Stock Dashboard")

company_key = st.sidebar.selectbox(
    "Select Company",
    list(COMPANIES.keys())
)

ticker = COMPANIES[company_key]

st.sidebar.markdown("---")

st.sidebar.write("**Project Technologies**")
st.sidebar.write("🐍 Python")
st.sidebar.write("📊 Machine Learning")
st.sidebar.write("🧠 Deep Learning")
st.sidebar.write("📈 Streamlit")
st.sidebar.write("🌐 Yahoo Finance")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Stock Price Prediction & Analytics Dashboard"
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("📊 Stock Price Prediction & Analytics Dashboard")

st.markdown(
    f'<p class="dashboard-subtitle">'
    f'Multi-Company Stock Analysis | {company_key} ({ticker})'
    f'</p>',
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# LOAD SELECTED COMPANY
# =========================================================

try:

    with st.spinner(
        f"Downloading {company_key} data and training models..."
    ):

        output = train_and_evaluate_models(ticker)

    data = output["data"]
    results = output["results"]

    best_model_name = output["best_model"]
    next_prediction = output["next_prediction"]

    latest_close = float(data["Close"].iloc[-1])
    latest_date = data.index[-1].strftime("%d %b %Y")

    highest_price = float(data["Close"].max())
    lowest_price = float(data["Close"].min())

    historical_records = len(data)

    prediction_difference = next_prediction - latest_close

    percentage_difference = (
        prediction_difference / latest_close
    ) * 100 if latest_close != 0 else 0

    latest_close_inr = convert_to_inr(latest_close, ticker)
    next_prediction_inr = convert_to_inr(next_prediction, ticker)

    difference_inr = convert_to_inr(
        prediction_difference, ticker
    )

    highest_price_inr = convert_to_inr(
        highest_price, ticker
    )

    lowest_price_inr = convert_to_inr(
        lowest_price, ticker
    )

    # =====================================================
    # DASHBOARD TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Market Trends",
        "🏢 Company Comparison",
        "🤖 Model Comparison",
        "🔍 Model Details"
    ])

    # =====================================================
    # TAB 1: OVERVIEW
    # =====================================================

    with tab1:

        st.subheader("📊 Stock Market Overview")

        st.caption(
            f"Latest available market data: {latest_date}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Closing Price",
            f"₹{latest_close_inr:,.2f}"
        )

        col2.metric(
            "Next Trading Day Estimate",
            f"₹{next_prediction_inr:,.2f}",
            f"{percentage_difference:.2f}%"
        )

        col3.metric(
            "Prediction Difference",
            f"₹{difference_inr:,.2f}"
        )

        st.markdown("---")

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Highest Closing Price",
            f"₹{highest_price_inr:,.2f}"
        )

        col5.metric(
            "Lowest Closing Price",
            f"₹{lowest_price_inr:,.2f}"
        )

        col6.metric(
            "Historical Records",
            f"{historical_records:,}"
        )

        st.markdown("---")

        st.subheader("🏆 Best Performing Model")

        st.success(
            f"Selected Model: {best_model_name}"
        )

        best_metrics = results[best_model_name]["Test"]

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "MAE",
            f"₹{convert_to_inr(best_metrics[0], ticker):,.2f}"
        )

        m2.metric(
            "RMSE",
            f"₹{convert_to_inr(best_metrics[1], ticker):,.2f}"
        )

        m3.metric(
            "R² Score",
            f"{best_metrics[2]:.4f}"
        )

        st.caption(
            "Best model is selected using validation RMSE. "
            "Test metrics are shown separately."
        )

        st.markdown("---")

        st.subheader("📈 Recent Stock Price")

        recent = data.tail(180)

        recent_fig = go.Figure()

        recent_fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent["Close"].apply(
                    lambda x: convert_to_inr(x, ticker)
                ),
                mode="lines",
                name="Closing Price"
            )
        )

        recent_fig.update_layout(
            title=f"{company_key} Recent Stock Prices",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            recent_fig,
            use_container_width=True
        )

    # =====================================================
    # TAB 2: MARKET TRENDS
    # =====================================================

    with tab2:

        st.subheader("📈 Candlestick Chart")

        candle_data = data.tail(250)

        candle_fig = go.Figure(
            data=[
                go.Candlestick(
                    x=candle_data.index,
                    open=candle_data["Open"].apply(
                        lambda x: convert_to_inr(x, ticker)
                    ),
                    high=candle_data["High"].apply(
                        lambda x: convert_to_inr(x, ticker)
                    ),
                    low=candle_data["Low"].apply(
                        lambda x: convert_to_inr(x, ticker)
                    ),
                    close=candle_data["Close"].apply(
                        lambda x: convert_to_inr(x, ticker)
                    ),
                    name="Stock Price"
                )
            ]
        )

        candle_fig.update_layout(
            title=f"{company_key} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_dark",
            height=500,
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(
            candle_fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 Moving Average Analysis")

        trend_data = data.copy()

        trend_data["MA20"] = (
            trend_data["Close"].rolling(20).mean()
        )

        trend_data["MA50"] = (
            trend_data["Close"].rolling(50).mean()
        )

        trend_fig = go.Figure()

        trend_fig.add_trace(
            go.Scatter(
                x=trend_data.index,
                y=trend_data["Close"].apply(
                    lambda x: convert_to_inr(x, ticker)
                ),
                mode="lines",
                name="Closing Price"
            )
        )

        trend_fig.add_trace(
            go.Scatter(
                x=trend_data.index,
                y=trend_data["MA20"].apply(
                    lambda x: convert_to_inr(x, ticker)
                ),
                mode="lines",
                name="20-Day Moving Average"
            )
        )

        trend_fig.add_trace(
            go.Scatter(
                x=trend_data.index,
                y=trend_data["MA50"].apply(
                    lambda x: convert_to_inr(x, ticker)
                ),
                mode="lines",
                name="50-Day Moving Average"
            )
        )

        trend_fig.update_layout(
            title="Stock Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            trend_fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 Trading Volume")

        volume_fig = go.Figure()

        volume_fig.add_trace(
            go.Bar(
                x=data.tail(180).index,
                y=data.tail(180)["Volume"],
                name="Trading Volume"
            )
        )

        volume_fig.update_layout(
            title="Historical Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_dark",
            height=400
        )

        st.plotly_chart(
            volume_fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📋 Recent Historical Data")

        display_data = data.tail(20).copy()

        for column in ["Open", "High", "Low", "Close"]:
            display_data[column] = display_data[column].apply(
                lambda x: convert_to_inr(x, ticker)
            )

        st.dataframe(
            display_data.round(2),
            use_container_width=True
        )

    # =====================================================
    # TAB 3: COMPANY COMPARISON
    # =====================================================

    with tab3:

        st.subheader("🏢 Compare Multiple Companies")

        st.write(
            "Compare the percentage movement of selected stocks "
            "over the last one year."
        )

        default_companies = [
            company_key,
            "Apple",
            "Microsoft"
        ]

        default_companies = list(dict.fromkeys(
            default_companies
        ))

        selected_companies = st.multiselect(
            "Select companies to compare",
            options=list(COMPANIES.keys()),
            default=default_companies
        )

        if len(selected_companies) >= 2:

            selected_tickers = [
                COMPANIES[name]
                for name in selected_companies
            ]

            with st.spinner("Loading comparison data..."):

                comparison_data = download_comparison_data(
                    tuple(selected_tickers)
                )

            if not comparison_data.empty:

                comparison_fig = go.Figure()

                for selected_ticker in comparison_data.columns:

                    comparison_fig.add_trace(
                        go.Scatter(
                            x=comparison_data.index,
                            y=comparison_data[selected_ticker],
                            mode="lines",
                            name=selected_ticker
                        )
                    )

                comparison_fig.update_layout(
                    title="Normalized Stock Performance (Starting Value = 100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price",
                    template="plotly_dark",
                    height=500
                )

                st.plotly_chart(
                    comparison_fig,
                    use_container_width=True
                )

                st.caption(
                    "Each company is normalized to 100 at its first available "
                    "date. This compares percentage movement, not actual prices."
                )

                latest_comparison = comparison_data.ffill().iloc[-1]

                comparison_summary = pd.DataFrame({
                    "Ticker": latest_comparison.index,
                    "Normalized Value": latest_comparison.values,
                    "Approx. Change (%)": (
                        latest_comparison.values - 100
                    )
                })

                st.subheader("📋 Comparison Summary")

                st.dataframe(
                    comparison_summary.round(2),
                    use_container_width=True
                )

            else:

                st.warning(
                    "Could not retrieve comparison data. Please try again."
                )

        else:

            st.info(
                "Please select at least two companies to compare."
            )

    # =====================================================
    # TAB 4: MODEL COMPARISON
    # =====================================================

    with tab4:

        st.subheader("🤖 Machine Learning Model Comparison")

        comparison_rows = []

        for model_name, model_result in results.items():

            val_mae, val_rmse, val_r2 = (
                model_result["Validation"]
            )

            test_mae, test_rmse, test_r2 = (
                model_result["Test"]
            )

            comparison_rows.append({
                "Model": model_name,
                "Validation MAE (INR)": convert_to_inr(
                    val_mae, ticker
                ),
                "Validation RMSE (INR)": convert_to_inr(
                    val_rmse, ticker
                ),
                "Validation R²": val_r2,
                "Test MAE (INR)": convert_to_inr(
                    test_mae, ticker
                ),
                "Test RMSE (INR)": convert_to_inr(
                    test_rmse, ticker
                ),
                "Test R²": test_r2
            })

        comparison_df = pd.DataFrame(
            comparison_rows
        )

        st.dataframe(
            comparison_df.round(4),
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 Test RMSE Comparison")

        rmse_fig = go.Figure()

        rmse_fig.add_trace(
            go.Bar(
                x=comparison_df["Model"],
                y=comparison_df["Test RMSE (INR)"],
                text=[
                    f"₹{x:,.2f}"
                    for x in comparison_df["Test RMSE (INR)"]
                ],
                textposition="auto",
                name="Test RMSE"
            )
        )

        rmse_fig.update_layout(
            title="Model Performance - Test RMSE",
            xaxis_title="Model",
            yaxis_title="RMSE (INR)",
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            rmse_fig,
            use_container_width=True
        )

        st.info(
            f"Best model based on validation RMSE: {best_model_name}"
        )

        st.markdown("---")

        st.subheader("📥 Download Model Results")

        csv_data = comparison_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Model Comparison CSV",
            data=csv_data,
            file_name=f"{ticker}_model_comparison.csv",
            mime="text/csv"
        )

    # =====================================================
    # TAB 5: MODEL DETAILS
    # =====================================================

    with tab5:

        st.subheader("🔍 Model Performance Details")

        st.markdown("""
        **1. Naive Baseline**

        Uses the previous trading day's closing price as the next day's estimate.

        **2. Linear Regression**

        Uses a linear relationship between the previous 60 closing prices
        and the next closing price.

        **3. Random Forest**

        Uses an ensemble of decision trees to estimate the next closing price.

        **4. LSTM**

        Uses a Long Short-Term Memory neural network to learn patterns
        in sequential stock-price data.
        """)

        st.markdown("---")

        st.subheader("📉 Actual vs Predicted")

        selected_model = st.selectbox(
            "Select Model",
            list(results.keys())
        )

        actual_values = output["y_test_actual"]

        predicted_values = output["test_predictions"][
            selected_model
        ]

        test_dates = output["test_dates"]

        actual_inr = [
            convert_to_inr(x, ticker)
            for x in actual_values
        ]

        predicted_inr = [
            convert_to_inr(x, ticker)
            for x in predicted_values
        ]

        prediction_fig = go.Figure()

        prediction_fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=actual_inr,
                mode="lines",
                name="Actual Price"
            )
        )

        prediction_fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=predicted_inr,
                mode="lines",
                name="Predicted Price"
            )
        )

        prediction_fig.update_layout(
            title=f"Actual vs Predicted - {selected_model}",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            prediction_fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("📊 Selected Model Test Metrics")

        selected_metrics = results[selected_model]["Test"]

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "MAE",
            f"₹{convert_to_inr(selected_metrics[0], ticker):,.2f}"
        )

        metric2.metric(
            "RMSE",
            f"₹{convert_to_inr(selected_metrics[1], ticker):,.2f}"
        )

        metric3.metric(
            "R² Score",
            f"{selected_metrics[2]:.4f}"
        )

        st.caption(
            "R² is a statistical evaluation score, not percentage accuracy."
        )

        st.markdown("---")

        st.subheader("📥 Download Historical Stock Data")

        historical_csv = data.to_csv().encode("utf-8")

        st.download_button(
            label="Download Historical Data CSV",
            data=historical_csv,
            file_name=f"{ticker}_historical_data.csv",
            mime="text/csv"
        )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.markdown("---")

    st.warning("""
    **⚠️ Important Disclaimer**

    This project is for educational purposes. Stock predictions are
    estimates, not guaranteed market prices or financial advice.
    Please do not make investment decisions based only on these
    model predictions.
    """)

    # =====================================================
    # FOOTER
    # =====================================================

    st.markdown("""
    <div style="text-align:center; padding:18px; color:#AABBDD;">
        <b>Stock Price Prediction & Analytics Dashboard</b><br>
        Built with Python | Machine Learning | LSTM | Streamlit<br>
        <small>Educational Project</small>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ERROR HANDLING
# =========================================================

except Exception as e:

    st.error(
        f"Model analysis failed: {str(e)}"
    )

    st.info(
        "Please check your internet connection, stock ticker and installed packages."
    )

    st.exception(e)
