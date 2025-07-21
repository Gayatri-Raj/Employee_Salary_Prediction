import streamlit as st
import pandas as pd"
import joblib

#loading the trained model
model=joblib.load("rf_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction",page_icon="💼 💵",layout="centered")

st.title("Employee Salary Prediction App")
st.markdown("Predicts whether a person's income is less than $50K or more than $50K, based on input")

st.sidebar.header("Enter employee details")
age=st.sidebar.slider("Age",18,70,35)
educational_num=st.sidebar.selectbox("Education Level",["Bachelors","Masters","PhD","HS-Grad","Assoc","Some-college"])
occupation=st.sidebar.selectbox("Job Role",["Prof-specialty","Craft-repair","Exec-managerial","Adm-clerical","Sales","Other-service",
                                            "Machine-op-inspct","others","Transport-moving", "Handlers-cleaners","Farming-fishing",
                                            "Tech-support","Protective-serv","Priv-house-serv","Armed-Forces"])  
hours-per-week=st.sidebar.slider("Hours-per-week",1,80,40)

#Read input data as a dataframe
df=pd.DataFrame({"age":[age],"education":[educational_num],"occupation":[occupation],"hours-per-week":[hours-per-week]}

#Write the input data
st.write("### Input Data")
st.write(df)

#Prediction
if st.button("Predict salary class:"):
        prediction=model.predict(df)
        st.success(f"Prediction:{prediction[0]}")

#Batch Prediction     
st.markdown("---") 
st.markdown("### Batch Prediction")
file=st.file_uploader("Upload a csv file for batch prediction",type=csv)

if file is not None:
    data=pd.read_csv(file)
    st.write("Uploaded file preview:", data.head())
    pred=model.predict(data)
    data["PredictedClass"]=pred
    st.write("Predictions:")
    st.write(data.head())
    csv=data.to_csv(index=False).encode('utf-8')
    st.download_button("Download predictions csv",csv,file_name='Predicted_Classes.csv',mime='text/csv')
