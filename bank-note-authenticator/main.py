import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())

#Check for missing values in the dataset
print(df.isnull().sum())

#iloc[:, :-1] means all rows and all columns except the last one (df.iloc[<row_selection>, <column_selection>])
#iloc[:, -1] means all rows and only the last column
#Features are the input variables (x) and labels are the output variable (y)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Features:")
print(x.head())
print("Labels:")
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)

model = RandomForestClassifier()
model.fit(x_train, y_train)
print("\nModel Trained Successfully")

y_pred = model.predict(x_test)
print("\nPredictions:")
print(y_pred)

accuracy_score = accuracy_score(y_test,y_pred)
print("\nAccuracy:", accuracy_score)

pickle_out = open("Classifiermodel.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

