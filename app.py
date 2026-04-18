import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import google.generativeai as genai

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

# =========================
# GEMINI SETUP
# =========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    df = df.drop_duplicates()
    return df

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# =========================
# TITLE
# =========================
st.title("Heart Disease Prediction and Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["Prediction", "Dashboard", "Records"])

# =========================
# TAB 1: PREDICTION + AI
# =========================
with tab1:
    st.subheader("Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", [0, 1])
        cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)

    with col3:
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("CA", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("Predict"):
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("High risk of heart disease")
            risk_label = "High risk"
        else:
            st.success("Low risk of heart disease")
            risk_label = "Low risk"

        st.info(f"Predicted probability: {probability:.2%}")

        with st.spinner("Generating AI explanation..."):
            prompt = f"""
A patient has the following medical data:
Age: {age}
Sex: {sex}
Chest Pain Type: {cp}
Resting Blood Pressure: {trestbps}
Cholesterol: {chol}
Fasting Blood Sugar > 120: {fbs}
Rest ECG: {restecg}
Max Heart Rate: {thalach}
Exercise Induced Angina: {exang}
Oldpeak: {oldpeak}
Slope: {slope}
CA: {ca}
Thal: {thal}

The machine learning model predicted: {risk_label} of heart disease
with probability {probability:.2%}.

Explain this result in simple language for a non-medical person.
Give short lifestyle advice.
Add a short warning that this is not a medical diagnosis.
"""

            try:
                explanation = ask_gemini(prompt)
                st.subheader("AI Explanation")
                st.write(explanation)
            except Exception as e:
                st.error(f"Gemini error: {e}")

# =========================
# TAB 2: DASHBOARD
# =========================
with tab2:
    st.subheader("Analytics Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Heart Disease = 1", int((df["target"] == 1).sum()))
    c3.metric("Heart Disease = 0", int((df["target"] == 0).sum()))

    fig1 = px.histogram(
        df,
        x="age",
        color="target",
        barmode="group",
        title="Age Distribution by Target"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(
        df,
        x="target",
        y="chol",
        color="target",
        title="Cholesterol by Target"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="age",
        y="thalach",
        color="target",
        title="Age vs Max Heart Rate"
    )
    st.plotly_chart(fig3, use_container_width=True)

    feature_importance = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig4 = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# TAB 3: RECORDS
# =========================
with tab3:
    st.subheader("Patient Records")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="heart_data.csv",
        mime="text/csv"
    )
