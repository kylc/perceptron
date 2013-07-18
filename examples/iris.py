import pandas as pd
import numpy as np
from perceptron import train, perceive

columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

train_data = pd.read_csv("data/iris.data", header=None)
train_data.columns = columns

class_map = {'Iris-setosa': 0, 'Iris-versicolor': 1}
p = train(train_data.ix[:,0:4].values, train_data.ix[:,4].map(class_map))

test_data = pd.read_csv("data/test_iris.data", header=None)
test_data.columns = columns

for vec, target in zip(test_data.ix[:,0:4].values, test_data.ix[:,4].map(class_map)):
    print "Perceived:", perceive(p, vec), "Actual:", target
