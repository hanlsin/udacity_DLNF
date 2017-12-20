from data_prep import data
from sklearn.linear_model import LinearRegression

bmi_life_model = LinearRegression().fit(
    data[['BMI']], data[['Life expectancy']])
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)
