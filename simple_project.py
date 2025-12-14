import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.model_selection import train_test_split

st.title("Student Performance ML App")

df = pd.read_csv("datasets/student_performance.csv")
df["Gender_value"] = df["gender"].map({"Male":0,"Female":1})
df["Pass/fail_value"] = df["pass_fail"].map({"Fail":0,"Pass":1})
df["admission_value"] = df["admission_status"].map({"No":0,"Yes":1})

st.dataframe(df.head(10))

columns = df.columns

target_cols = ["final_grade","Pass/fail_value","admission_value"]

target_c = st.selectbox("Select Target variable",target_cols)

x_features_col = []

for c in columns:
    if c not in target_cols:
        x_features_col.append(c)


x_featurecolumn = st.multiselect("Select X features",x_features_col)
xfeature = df[x_featurecolumn]
ytarget = df[[target_c]]


if target_c == "final_grade":
    st.write("Linear regression will be applied..")

else:
    algo_name = st.selectbox("Select Algorithms",["KNN","Decision Tree","Logistic","Random forest"])
        
if st.button("Train model"):
    if len(x_featurecolumn)==0:
         st.error("Please select any X_features")

    else:
        if target_c == "final_grade":
           model = LinearRegression()
        else:
          if algo_name == "Logistic":
            model = LogisticRegression()
          elif algo_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
          elif algo_name == "Decision Tree":
            model = DecisionTreeClassifier()
          else:
            model = RandomForestClassifier(n_estimators=100)
    xtrain,xtest,ytrain,ytest = train_test_split(xfeature,ytarget,train_size=0.8,random_state=3)

    model.fit(xtrain,ytrain)

    ypred_test = model.predict(xtest)
    ypred_train = model.predict(xtrain)

    if target_c == "final_grade":
       acc_train = round(model.score(xtrain,ytrain)*100)
       acc_test = round(model.score(xtest,ytest)*100)

       st.success(f"Training Acccuracy:{acc_train}")
       st.success(f"Testing Acccuracy:{acc_test}")

    else:
       acc_train = round(accuracy_score(ytrain,ypred_train)*100)
       acc_test = round(accuracy_score(ytest,ypred_test)*100)

       st.success(f"Training Acccuracy:{acc_train}")
       st.success(f"Testing Acccuracy:{acc_test}")

    

    

    





