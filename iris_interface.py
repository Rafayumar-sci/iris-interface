import joblib
import streamlit as st

st.title("IRIS flower specie predictor")

st.write("""
This model helps you predict specie of iris flower based on its features
Please input following parameters    
""")

sepal_length = st.number_input(
    "sepal length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input(
    "sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input(
    "petal length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input(
    "petal width (cm)", min_value=0.0, max_value=15.0, value=12.0)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

with open("decision_iris.joblib", 'rb')as model_file:
    model = joblib.load(model_file)

with open("iris_encoder.joblib", "rb")as le_file:
    label_encoder = joblib.load(le_file)

if st.button("predict species"):
    with st.spinner("predicting..."):
        prediction = model.predict(input_data)
        species = label_encoder.inverse_transform(prediction)
        st.success(f"the predicted species is: {species[0]}")
