import pandas as pd

data_path = './udacity_DLNF/Part1.NeuralNetworks/L5.Regression/bmi_n_life_exp.csv'
data = pd.read_csv(data_path)

print(data.head())
