#Importing the necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Loading the dataset
data=pd.read_csv("adult.csv")

data=data[(data['age']>=17) &(data['age']<=75)]

#Resolving data redundancy
data=data[data["workclass"]!="Never-worked"]
data=data[data["workclass"]!="Without-pay"]

data=data[data["education"]!="Preschool"]
data=data[data["education"]!="1st-4th"]
data=data[data["education"]!="5th-6th"]

data=data.drop(columns=["education"])

#Handling missing values
data.workclass.replace({"?":"others"},inplace=True)
data.occupation.replace({"?":"others"},inplace=True)
data['native-country'].replace({"?":"others"},inplace=True)

#Encoding
encoder=LabelEncoder()
data["workclass"]=encoder.fit_transform(data["workclass"])
data["marital-status"]=encoder.fit_transform(data["marital-status"])
data["occupation"]=encoder.fit_transform(data["occupation"])
data["relationship"]=encoder.fit_transform(data["relationship"])
data["race"]=encoder.fit_transform(data["race"])
data["gender"]=encoder.fit_transform(data["gender"])
data["native-country"]=encoder.fit_transform(data["native-country"])

#Splitting the data
X=data.drop(columns=["income"])
y=data["income"]

#Normalizing the data
scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23,stratify=y)

#ML Algorithm
result={}
algorithm={'knn':KNeighborsClassifier(),'random forest':RandomForestClassifier(n_estimators=300),'logistic regression':LogisticRegression(),'SVM':SVC()}
for model,clf in algorithm.items():
    name=model
    model=clf
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print(name," accuracy:",acc)
    result[name]=acc
    print(classification_report(y_test,y_pred))
    
#Boosting Algorithm
import lightgbm as lgb
y_train1 = y_train.replace({'<=50K': 0, '>50K': 1})
y_test1 = y_test.replace({'<=50K': 0, '>50K': 1})

feature_names = ["age","workclass","fnlwgt","educational-num","marital-status","occupation","relationship","race","gender","capital-gain","capital-loss","hours-per-week","native-country"]


train_data = lgb.Dataset(X_train, label=y_train1, feature_name=feature_names)
test_data = lgb.Dataset(X_test, label=y_test1, feature_name=feature_names)
params = {
 'objective': 'binary',
 'metric': 'binary_logloss',
 'is_unbalance': True,
 'boosting_type': 'gbdt',
 'verbosity': -1
 }
model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)
 #Model Evaluation and accuracy
y_pred_prob=model.predict(X_test)
y_pred1=(y_pred_prob>=0.5).astype(int)
acc1=accuracy_score(y_test1,y_pred1)
print("LightGBM: \nAccuracy:",acc1)
result["LightGBM"]=acc1

#Model Comparison
plt.bar(result.keys(),result.values())
plt.title("Model Performance")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()

#Obtaining the best model
for model,acc in result.items():
    print(f"{model}:{acc:.2f}")

best_model_name = max(result, key=result.get)
best_model = algorithm[best_model_name]

print(f"\nBest model: {best_model_name} with score: {result[best_model_name]:.2f}")
joblib.dump(best_model,'rf_model.pkl')
print("Saved the best model as rf_model.pkl")


import streamlit as st
import pandas as pd
import joblib

#loading the trained model
model=joblib.load("rf_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction",page_icon="💼 💵",layout="centered")

st.title("Employee Salary Prediction App")
st.markdown("Predicts whether a person's income is less than $50K or more than $50K, based on input")

st.sidebar.header("Enter employee details")
age=st.sidebar.slider("Age",18,70,35)
education=st.sidebar.selectbox("Education Level",["Bachelors","Masters","PhD","HS-Grad","Assoc","Some-college"])
occupation=st.sidebar.selectbox("Job Role",["Prof-specialty","Craft-repair","Exec-managerial","Adm-clerical","Sales","Other-service",
                                            "Machine-op-inspct","others","Transport-moving", "Handlers-cleaners","Farming-fishing",
                                            "Tech-support","Protective-serv","Priv-house-serv","Armed-Forces"])  
hours_per_week=st.sidebar.slider("Hours-per-week",1,80,40)

#Read input data as a dataframe
df=pd.DataFrame({"age":[age],"education":[education],"occupation":[occupation],"hours-per-week":[hours_per_week]})

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
