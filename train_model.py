import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# 📊 Dữ liệu ví dụ
data = pd.DataFrame({
    'temperature': [30, 32, 28, 35, 31, 29, 33],
    'rain': [0, 0, 1, 0, 1, 1, 0],
    'promotion': [1, 0, 0, 1, 1, 0, 1],
    'weekday': [1, 1, 0, 1, 0, 0, 1],
    'sales': [120, 100, 80, 140, 110, 90, 135]
})

# 🎯 Tách feature và target
X = data[['temperature', 'rain', 'promotion', 'weekday']]
y = data['sales']

# ✂️ Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Huấn luyện mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Lưu mô hình
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Mô hình đã được huấn luyện và lưu vào model.pkl")

