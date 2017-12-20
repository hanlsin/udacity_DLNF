import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set pandas display settings
#pd.set_option('display.max_row', 5)
#pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 1000)

data_path = "./udacity_DLNF/Part1.NeuralNetworks/Proj_1/first-neural-network/Bike-Sharing-Dataset/hour.csv"
rides = pd.read_csv(data_path)
# data is between 2011-01-01 ~ 2012-12-31 by "dteday"
# - workingday = weekend is not workingday (0), the others are (1)
# - temp = temperature
# - atemp = ??
# - hum = humidity
# print(rides.head())
# print(rides.tail())
# print("distinct value of 'workingday': {0}".format(
#     rides['workingday'].unique()))
# print("distinct value of 'atemp'??   : {0}".format(
#     rides['atemp'].unique()))

# show plot of 10 days (2011-01-01 ~ 2011-01-10)
#rides[:24 * 10].plot(x='dteday', y='cnt')
# plt.show()

##################################
# Dummy variables
#####
# 특정 변수의 값을 바이너리(0 또는 1) 변수 매트릭스로 변경하는 작업.
# 예를 들어, season 변수의 값은 1~4 사이의 값을 가지는데 이를 바이너리 매트릭스로 변경하면,
#    season --> (season1, season2, season3, season4)
#       1   --> (1, 0, 0, 0)
#       2   --> (0, 1, 0, 0)
#       3   --> (0, 0, 1, 0)
#       4   --> (0, 0, 0, 1)
# 이렇게 변경된다.
# rides[:].plot(x='dteday', y='season')
# rides['season'].value_counts().plot(kind='bar')
# dummy_season = pd.get_dummies(rides['season'], prefix='season')
# dummy_season = pd.concat([rides['dteday'], dummy_season], axis=1)
# print(dummy_season.head())
# dummy_season[:].plot(x='dteday', y='season_1')
# dummy_season[:].plot(x='dteday', y='season_2')
# dummy_season[:].plot(x='dteday', y='season_3')
# dummy_season[:].plot(x='dteday', y='season_4')
# plt.show()

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    #print("distinct value of '{0}': {1}".format(each, rides[each].unique()))
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
# print(rides.head())

fields_to_drop = ['instant', 'dteday', 'weekday', 'workingday', 'atemp']
fields_to_drop.extend(dummy_fields)
data = rides.drop(fields_to_drop, axis=1)
# print(data.head())

##################################
# Scaling target variables
#####
# Training을 쉽게 하기 위해서 변수 값들을 정규화(normalization) 한다.
# 다시 말해, 변수들의 값을 수정(shift and scale)해서 평균이 0이 되도록 하고,
# 각 값의 크기를 1보다 작은 값을 같도록 규격화 시키는 것이다.
# (ref) http://adnoctum.tistory.com/184
# (ret) http://www.ktword.co.kr/abbr_view.php?nav=2&id=582&m_temp1=933
#
# 하지만, 설명에는 'standardize' 라고 표현한다.
# "That is, we'll shift and scale the variables
# such that they have zero mean and a standard deviation of 1."
# 이는 변수들 값의 일반 정규분포를 표준정규분포로 바꾸는 작업이다. (standardization, 표준화)
# 정규 분포는 평균이 0이고, 표준 편차(standard deviation) 값이 1이다.
# (ref) http://math7.tistory.com/47
# (ref) https://namu.wiki/w/%EC%A0%95%EA%B7%9C%EB%B6%84%ED%8F%AC
#
# Normalization vs. Standardzation
# - normalization:
# "normalizing your data will certainly scale the “normal” data to a very small interval."
# - standarization:
# "When using standardization, your new data aren’t bounded (unlike normalization)."
# 평균 값을 0으로 만드는 것은 동일하다.
#
# (ref) http://www.dataminingblog.com/standardization-vs-normalization/
# (ref) https://stats.stackexchange.com/questions/10289/whats-the-difference-between-normalization-and-standardization

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# 각 변수의 평균(mean)과 표준편차(stdev) 값을 저장해 두고, 이후 예측 값을 계산할 때 사용한다.
scaled_features = {}
for each in quant_features:
    mean, stdev = data[each].mean(), data[each].std()
    # print("\n[mean, stdev] value of '{0}': [{1}, {2}]".format(
    #     each, mean, stdev))
    scaled_features[each] = [mean, stdev]
    # rescale data by standardzation
    # print("BEFORE: {1} <= '{0}' <= {2}".format(
    #     each, data[each].min(), data[each].max()))
    data.loc[:, each] = (data[each] - mean) / stdev
    # print("AFTER : {1} <= '{0}' <= {2}".format(
    #     each, data[each].min(), data[each].max()))

##################################
# Splitting the data into training, testing, and validation sets
#####
# - test set : 가장 최근 21일 데이터
#              network를 train한 후 이 데이터로 prediction한 후 비교할 것임.
#              가장 최근 데이터에 대한 prediction을 높이는 것이
#              이 후 데이터에 대한 정확도가 높을 확률이 높다.
# - validation set : network의 정확성을 판단하기 위해 test set을 제외한 최근 데이터 일부
# - train set : network를 교육시킬 자료는 가장 오래된 것들로
#               test set, validation set을 제외한 대부분 데이터
#
# test set과 validation set의 차이는 뭐지?
# https://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set
# 결국 test set은 training 중에 볼 수 있는 데이터가 아니기 때문에 최종 결과 보고서(?)를 위한 데이터라는 것.

# separrate the data into features and targets
# feature들의 값을 이용해 target의 결과를 얻어낸다.
target_fields = ['cnt', 'casual', 'registered']

# test set
test_data = data[-21 * 24:]
test_targets = test_data[target_fields]
test_features = test_data.drop(target_fields, axis=1)
# print(test_data.head())
# print(test_data.tail())

# remove test set in the data
data = data[:-21 * 24]

# train set
train_data = data[:-60 * 24]
train_targets = train_data[target_fields]
train_features = train_data.drop(target_fields, axis=1)
# print(train_data.head())
# print(train_data.tail())

# validation set
val_data = data[-60 * 24:]
val_targets = val_data[target_fields]
val_features = val_data.drop(target_fields, axis=1)
# print(val_data.head())
# print(val_data.tail())

##################################
# the network in my_answers.py
#####
from my_answers import NeuralNetwork

##################################
# unit tests
#####
import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
# print(inputs)
# print(inputs.shape)
targets = np.array([[0.4]])
# print(targets)
# weight: input to hidden
test_w_i_h = np.array([
    [0.1, -0.2],
    [0.4, 0.5],
    [-0.3, 0.2]
])
# print(test_w_i_h)
# weight: hidden to outpus
test_w_h_o = np.array([
    [0.3],
    [-0.1]
])
# print(test_w_h_o)


class TestMethods(unittest.TestCase):
    def test_data_loaded(self):
        self.assertTrue(isinstance(rides, pd.DataFrame))

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        sigmoid = 1 / (1 + np.exp(-0.5))
        self.assertTrue(network.activation_function(0.5) == sigmoid)

    def test_train(self):
        network = NeuralNetwork(3, 2, 1, 0.5)

        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        # print(network.weights_input_to_hidden)
        # print(network.weights_hidden_to_output)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


suite = unittest.TestSuite()
tests = unittest.TestLoader().loadTestsFromTestCase(TestMethods)
suite.addTests(tests)
unittest.TextTestRunner().run(suite)
