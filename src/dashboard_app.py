
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Inventory Shortage Dashboard")

df = pd.read_csv("data/features/inventory_features.csv")
product_ids = df["product_id"].unique()
store_ids = df["store_id"].unique()

selected_store = st.selectbox("Select Store", store_ids)
selected_product = st.selectbox("Select Product", product_ids)

filtered_df = df[(df["store_id"] == selected_store) & (df["product_id"] == selected_product)]

st.write("Historical Stock Level and Shortage Events")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(filtered_df["date"], filtered_df["stock_level"], label="Stock Level", color="blue")
shortage_dates = filtered_df[filtered_df["shortage_event"] == 1]["date"]
for sd in shortage_dates:
    ax.axvline(pd.to_datetime(sd), color="red", linestyle="--", alpha=0.5)
ax.set_xlabel("Date")
ax.set_ylabel("Stock Level")
ax.legend()
st.pyplot(fig)
