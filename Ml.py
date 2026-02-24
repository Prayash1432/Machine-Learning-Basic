import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Test.csv")  # Reading the dataset
df.columns = df.columns.str.strip()  # Remove Extra spaces from column name]

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area, df.price, color='red', marker='+')

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
reg.predict([[3000]])

d = pd.read_csv("Check.csv")
p = reg.predict(d[['area']])
d['prices'] = p
d.to_csv("Checking.csv", index=False)

plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
