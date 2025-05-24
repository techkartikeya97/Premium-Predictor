import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

nav = st.sidebar.radio("Navigation", ["About", "Predict"])
df = pd.read_csv('/home/kartikeyas1997/projects/Insurance Predictor/insurance.csv')

if nav == "About":
    st.title("Health Insurance Premium Predictor")
    st.text(" ")
    st.text(" ")
    st.text("Predict Your Health Insurance Premium")
    st.image('health_insurance.jpeg', width=600)

df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

x = df.drop(columns='charges', axis=1)
y = df['charges']

# Split dataset to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)

# Calculate accuracy on test set
y_pred = rfr.predict(x_test)
accuracy = r2_score(y_test, y_pred)

if nav == "About":
    st.markdown(f"### Model Accuracy: **{accuracy:.2%}**")

if nav == "Predict":
    st.title("Enter Details")

    age = st.number_input("Age: ", step=1, min_value=0)
    sex = st.radio("Sex", ("Male", "Female"))
    s = 0 if sex == "Male" else 1

    bmi = st.number_input("BMI: ", min_value=0)
    children = st.number_input("Number of children: ", step=1, min_value=0)
    smoke = st.radio("Do you smoke", ("Yes", "No"))
    sm = 0 if smoke == "Yes" else 1

    region = st.selectbox('Region', ('SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'))
    reg = {"SouthEast": 0, "SouthWest": 1, "NorthEast": 2, "NorthWest": 3}[region]

    if st.button("Predict"):
        premium = rfr.predict([[age, s, bmi, children, sm, reg]])[0]
        st.subheader("Predicted Premium")
        st.markdown(f"The estimated insurance premium is **${premium:.2f}**")
