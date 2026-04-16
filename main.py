import pandas as pd

df = pd.read_csv("loan_approval_1000.csv")

#data exploration
print(df.head())
print(df.isnull().sum())

mean = df["Income"].mean()
print("Mean Income:", mean)

mode = df["Employment_Type"].mode()[0]
print("Mode Employment Type : ", mode)

#filling missing values with mean and 
df["Income"] = df["Income"].fillna(mean)
df["Employment_Type"] = df["Employment_Type"].fillna(mode)

print(df.isnull().sum())