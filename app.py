import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("Passenger Survival Chance in the Titanic Journey")

pclass= st.slider("Enter the Passenger class for the user",1,3)
sex = st.selectbox("Enter the Passenger Gender",["female","male"])

sibSp= st.slider("Enter the Sibling Spouse of the user",1,8)

parch=st.slider("Enter the Passenger Parents and child for the user",1,6)
fare=st.number_input("Enter the Fare of the passenger")
embarked = st.selectbox("Enter the Passenger station from where they started the journey",['Queenstown','Chebourg','Southampton'])

data=pd.DataFrame([{
  'Pclass':pclass, 
  'Sex':sex, 
  'SibSp':sibSp, 
  'Parch':parch, 
  'Fare':fare,
  'Embarked':embarked}])

if st.button("Data"):
   st.write(data)

model = load_model("model.h5")


with open("label.pkl","rb")as f:
   label=pickle.load(f)

with open("ohe.pkl","rb")as f:
   ohe=pickle.load(f)

with open("scaler.pkl","rb")as f:
   scaler=pickle.load(f)

data["Sex"] = label.transform(data["Sex"])
embarked=ohe.transform(data[["Embarked"]])

embarked_df=pd.DataFrame(embarked,columns=ohe.get_feature_names_out())

data=pd.concat([data.drop(columns=["Embarked"]),embarked_df],axis=1)

data[['Pclass', 'SibSp', 'Parch', 'Fare']]=scaler.transform(data[['Pclass', 'SibSp','Parch', 'Fare']])

y=model.predict(data)

y_predict = y[0][0]

print(y_predict)
def chance(y_predict):
   if y_predict > 0.5 and y_predict < 0.7 :
      return "Person is likely to survive."
   elif y_predict > 0.7 :
      return "Person is more likely or has many chances of survive."
   else:
      return "Person will not survive."

if st.button("Predict Survival Chance"):
   st.write("Probability of Passenger Survival Chance",y_predict)
   st.write(chance(y_predict))
