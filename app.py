# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# Load model
model = pickle.load(open("model.pkl", "rb"))

def main():
    st.title("Dự đoán doanh số bán trà sữa")

    st.write("### Nhập thông tin:")
    weather = st.selectbox("Thời tiết", ["Nắng", "Mưa", "Âm u"])
    day = st.selectbox("Ngày trong tuần", ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"])
    promotion = st.selectbox("Có khuyến mãi?", ["Có", "Không"])

    # Encode input manually
    weather_map = {"Nắng": 0, "Mưa": 1, "Âm u": 2}
    day_map = {"Thứ 2": 0, "Thứ 3": 1, "Thứ 4": 2, "Thứ 5": 3, "Thứ 6": 4, "Thứ 7": 5, "Chủ nhật": 6}
    promo_map = {"Không": 0, "Có": 1}

    X_input = np.array([[weather_map[weather], day_map[day], promo_map[promotion]]])

    if st.button("Dự đoán doanh số"):
        result = model.predict(X_input)
        st.success(f"Dự đoán doanh số: {int(result[0])} ly")

if __name__ == '__main__':
    main()

