import streamlit as st
from datetime import datetime, timedelta
from utils import load_data, plot_data

st.title("📈 Stock Price Predictor")

# Sidebar inputs
ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL").upper()
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.today())

if start_date >= end_date:
    st.error("⚠️ End date must be after start date.")
else:
    df = load_data(ticker, start_date, end_date)

    if df.empty:
        st.warning("No data available for this ticker or time range.")
    else:
        st.subheader(f"Showing Closing Price for {ticker}")
        st.dataframe(df.tail())
        st.line_chart(df)  # Streamlit works if df = Date index + single numeric column
        st.pyplot(plot_data(df))
