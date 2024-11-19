import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#Page configuration
st.set_page_config(
    page_title="Simple Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
#title of the app
st.title("Simple Prediction App")
df=sns.load_dataset("iris")
#inputwidgets
st.sidebar.subheader("Input featuring")
sepal_length=st.sidebar.slider("Sepal Length",4.3,7.9,5.8)
sepal_width= st.sidebar.slider("Sepal width",2.0,4.4,3.1)
petal_length=st.sidebar.slider("Petal Length",1.0,6.9,3.8)
petal_width=st.sidebar.slider("Petal Width",0.1,2.5,1.2)

#separating the dependent and indepentent variables
X=df.drop(columns="species",axis=1)
y=df["species"]

#data splitng
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#model Training
rf=RandomForestClassifier(max_depth=2, max_features=4,n_estimators=200,random_state=42)
rf.fit(X_train,y_train)

#apply for prediction
y_pred=rf.predict([[sepal_length,sepal_width,petal_length,petal_width]])

#print EDA
st.subheader("Brief EDA")
st.write("The data is grouped by the class and variable mean is computed for each class")
groupby_species_mean=df.groupby(["species"]).mean()

from typing import TypedDict

class MyDict(TypedDict):
    key: str

st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)
#Print inout features
Input_feature=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                           columns=["sepal_length","sepal_width","petal_length","petal_width"])
st.write(Input_feature)

#Print prediction output
st.subheader("output")
st.metric("Predicted class",y_pred[0])