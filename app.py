import streamlit as st

st.set_page_config(
    page_title="Arthritis Tracker",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Arthritis Symptom Tracker")
st.markdown("Track daily joint pain and get smart, personalized advice using AI.")
st.markdown("---")

import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import requests 
API_KEY = "8d0deabf47db332da6daf522fff50856"
import os


st.title("🦴 Arthritis Tracker App")

st.markdown("Log your daily symptoms and track your progress.")

# Input form
with st.form("log_form"):
    st.subheader("📝 Daily Symptom Log")
    date = st.date_input("Date")
    city = st.text_input("City")

    col1, col2 = st.columns(2)
    with col1:
        pain = st.slider("Pain Level", 0, 10)
        mood = st.selectbox("Mood", ["😊 Happy", "😐 Neutral", "😞 Sad"])
    with col2:
        stiffness = st.slider("Stiffness Level", 0, 10)
        activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])

    submitted = st.form_submit_button("Save Entry")
    if submitted:
    st.success("✅ Entry saved! Your data is being used to improve tomorrow’s predictions.")


# Weather fetch function
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temp, humidity
    else:
        return None, None

# Save to CSV
if submitted:
    temp, humidity = get_weather(city)
    if temp is None:
        st.error("⚠️ Could not fetch weather. Check city name or API key.")
    else:
        new_entry = {
            "Date": entry_date,
            "City": city,
            "Temperature": temp,
            "Humidity": humidity,
            "Pain": pain_level,
            "Stiffness": stiffness,
            "Activity": activity,
        }

        
# 🎯 Add AI predictions and advice before saving
predicted_pain = prediction if 'prediction' in locals() else ""
predicted_stiffness = prediction_stiff if 'prediction_stiff' in locals() else ""

# Convert advice into text
if predicted_pain == "":
    pain_advice = ""
elif predicted_pain < 4:
    pain_advice = "Stay active"
elif predicted_pain <= 6:
    pain_advice = "Balance activity and rest"
else:
    pain_advice = "Rest and recovery"

if predicted_stiffness == "":
    stiffness_advice = ""
elif predicted_stiffness < 4:
    stiffness_advice = "Light movement or yoga"
elif predicted_stiffness <= 6:
    stiffness_advice = "Warm shower and short stretches"
else:
    stiffness_advice = "Avoid repetitive motion, go slow"

# 📦 Build the full record
new_entry = {
    "Date": date,
    "City": city,
    "Pain": pain,
    "Stiffness": stiffness,
    "Mood": mood,
    "Activity": activity,
    "Temperature": temp,
    "Humidity": humidity,
    "Predicted_Pain": predicted_pain,
    "Predicted_Stiffness": predicted_stiffness,
    "Pain_Advice": pain_advice,
    "Stiffness_Advice": stiffness_advice
}

# 📄 Save to CSV
try:
    df = pd.read_csv("arthritis_log.csv")
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
except FileNotFoundError:
    df = pd.DataFrame([new_entry])

df.to_csv("arthritis_log.csv", index=False)
st.success("✅ Entry with predictions saved!")

# Show past entries
st.markdown("### 📊 Past Symptom Log")
try:
    df = pd.read_csv("arthritis_log.csv")
    st.dataframe(df[["Date", "City", "Temperature", "Humidity", "Pain", "Stiffness", "Activity"]])


    st.line_chart(df.set_index("Date")[["Pain", "Stiffness"]])
except FileNotFoundError:
    st.info("No entries yet.")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.markdown("### 🔮 Tomorrow’s Predictions")

try:
    df = pd.read_csv("arthritis_log.csv")

    # Preprocessing
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.dayofyear

    # Encode activity level (Low=0, Medium=1, High=2)
    le = LabelEncoder()
    df["ActivityEncoded"] = le.fit_transform(df["Activity"])

    # Train a simple model
    X = df[["Day", "ActivityEncoded"]]
    y = df["Pain"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict tomorrow
    tomorrow_day = df["Day"].max() + 1
    user_activity = st.selectbox("Select activity level for prediction", ["Low", "Medium", "High"])
    activity_encoded = le.transform([user_activity])[0]

    predicted_pain = model.predict([[tomorrow_day, activity_encoded]])[0]
    st.success(f"🧠 Predicted pain level for tomorrow: **{predicted_pain:.2f}**")

except Exception as e:
    st.info("Add at least a few entries to see predictions.")
import requests

import requests

API_KEY = "8d0deabf47db332da6daf522fff50856"
import matplotlib.pyplot as plt

st.markdown("### 📊 Weather vs Symptom Visualizations")

try:
    df = pd.read_csv("arthritis_log.csv")

    # Plot Pain vs Temperature
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Temperature"], df["Pain"], color="red")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Pain Level")
    ax1.set_title("Pain vs Temperature")
    st.pyplot(fig1)

    # Plot Pain vs Humidity
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["Humidity"], df["Pain"], color="blue")
    ax2.set_xlabel("Humidity (%)")
    ax2.set_ylabel("Pain Level")
    ax2.set_title("Pain vs Humidity")
    st.pyplot(fig2)

except Exception as e:
    st.info("⚠️ Please enter some symptom data to view visualizations.")

from sklearn.linear_model import LinearRegression
import numpy as np

st.markdown("### 🤖 AI Prediction: Tomorrow's Pain Level")

try:
    df = pd.read_csv("arthritis_log.csv")
    df = df.dropna()

    # Prepare training data
    X = df[["Temperature", "Humidity", "Pain"]].iloc[:-1]
    y = df["Pain"].shift(-1).dropna()

    model = LinearRegression()
    model.fit(X, y)

    latest = df[["Temperature", "Humidity", "Pain"]].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(latest)[0]

    st.success(f"🔮 Predicted pain level for tomorrow: {prediction:.2f}")

    # 💡 Smart recommendation
    st.markdown("### 💡 Recommended Activity Level")

    if prediction < 4:
        st.info("✅ Great day to stay active! Try going for a short walk or light exercise.")
    elif 4 <= prediction <= 6:
        st.warning("⚠️ Moderate pain expected. Balance activity with rest.")
    else:
        st.error("🔴 High pain predicted. Prioritize rest, gentle stretches, and hydration.")

except Exception as e:
    st.info("Not enough data for prediction. Please log at least 2 days.")
st.markdown("### 🤖 AI Prediction: Tomorrow's Stiffness Level")

try:
    # Reuse the same CSV
    df = pd.read_csv("arthritis_log.csv")
    df = df.dropna()

    # Features for stiffness prediction
    X_stiff = df[["Temperature", "Humidity", "Stiffness"]].iloc[:-1]
    y_stiff = df["Stiffness"].shift(-1).dropna()

    model_stiff = LinearRegression()
    model_stiff.fit(X_stiff, y_stiff)

    latest_stiff = df[["Temperature", "Humidity", "Stiffness"]].iloc[-1].values.reshape(1, -1)
    prediction_stiff = model_stiff.predict(latest_stiff)[0]

    st.success(f"🔮 Predicted stiffness level for tomorrow: {prediction_stiff:.2f}")

    # Recommendation based on stiffness
    st.markdown("### 💡 Stiffness Management Suggestion")
    if prediction_stiff < 4:
        st.info("✅ Flexibility should be okay. Light yoga or movement may help keep joints loose.")
    elif 4 <= prediction_stiff <= 6:
        st.warning("⚠️ Mild stiffness expected. Consider warm showers and short stretches.")
    else:
        st.error("🔴 High stiffness likely. Minimize repetitive motions and take it slow in the morning.")

except Exception as e:
    st.info("Not enough data for stiffness prediction.")

st.markdown("### 📊 Past Entries")
st.dataframe(pd.read_csv("arthritis_log.csv"))

import matplotlib.pyplot as plt

if not df.empty:
    st.markdown("### 📈 Pain & Stiffness Over Time")
    fig, ax = plt.subplots()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    ax.plot(df["Date"], df["Pain"], label="Pain")
    ax.plot(df["Date"], df["Stiffness"], label="Stiffness")
    ax.set_ylabel("Level")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)
st.sidebar.title("🔍 Filter Logs")
filter_mood = st.sidebar.selectbox("Mood Filter", ["All", "😊 Happy", "😐 Neutral", "😞 Sad"])
# Then filter df accordingly before displaying it.

