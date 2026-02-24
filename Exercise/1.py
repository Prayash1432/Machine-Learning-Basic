import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("canada_per_capita_income.csv")
df.columns = df.columns.str.strip()

reg = LinearRegression()
reg.fit(df[['year']], df[['per capita income (US$)']])

d = pd.read_csv("y.csv")
p = reg.predict(d[['year']])
d['per capita income (US$)'] = p
p
d.to_csv("canada_per_capita_income_predicted.csv", index=False)

plt.scatter(df['year'], df[['per capita income (US$)']],
            color='red', marker='+')
plt.plot(d.year, d['per capita income (US$)'], color='blue')
