from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from pandas import DataFrame, read_csv
import pandas as pd
from collections import Counter
from sklearn import preprocessing
import numpy as np
import random

##Collaborated with Jared and Robbie


class Dataset:
    def __init__(self, csv, test_size, random_state, convert_nominal=False):
        self.ds = pd.read_csv(csv, header=None)
        ds_column_len = len(self.ds.columns)

        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.std_train = []
        self.std_test = []

        self.data = self.ds.loc[:, : ds_column_len - 2]
        self.targets = self.ds[ds_column_len - 1]



        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                self.data, self.targets, test_size=test_size, random_state=random_state)
        self.std_train = [0] * int(len(self.data_train))
        self.std_test = [0] * int(len(self.data_test))

        print("The dataframe before: \n", self.ds)
        if convert_nominal is True:
            self.convert_numerical_data(self.ds)
        else:
            self.convert_nominal_data(self.ds)
            self.standardize_data()
        print("The dataframe after: \n", self.ds)
        # This function uses a zscore to standardize the data

        if not convert_nominal:
            self.standardize_data()


        print("train data and targets", self.data_train, self.target_train)

    # This function turns nominal data into number values that are not scaled
    def convert_nominal_data(self, dataset):
        le = preprocessing.LabelEncoder()
        num_col = len(dataset.columns)

        for i in range(0, num_col):
            has_string = False
            le.fit(dataset[i])
            list_of_classes = list(le.classes_)
            for a_class in list_of_classes:
                if isinstance(a_class, str):
                    has_string = True
            if has_string:
                dataset[i] = le.transform(dataset[i])

    # this function turns numeric data into generalized data (ex. 1-2 = 0, 3-4 = 1, 5-6 = 2)
    def convert_numerical_data(self, dataset):
        num_col = len(dataset.columns)
        le = preprocessing.LabelEncoder()

        for i in range(0, num_col):
            nominal_col = False
            le.fit(dataset[i])
            list_of_classes = list(le.classes_)
            for a_class in list_of_classes:
               if isinstance(a_class, str):
                  nominal_col = True
                  break
            if nominal_col:
               continue
            else:
               dataset[i] = pd.qcut(dataset[i], 4, labels=False)

    def standardize_data(self):
        # fill std_train with standardized values
        self.std_train = (self.data_train - self.data_train.mean()) / self.data_train.std()
        # fill std_test with standardized values
        self.std_test = (self.data_test - self.data_train.mean()) / self.data_train.std()

class NeuralNetwork:
    def __init__(self,learning_rate, data, targets, inputs, bias=-1):
        self.learning_rate = learning_rate
        self.data = data
        self.targets = targets
        self.inputs = inputs
        self.labels = np.unique(targets)
        num_inputs = len(self.inputs.columns)
        self.input_list = []
        self.neurons = []
        for label in self.labels:
            new_neuron = self.Neuron(num_inputs, label)
            self.neurons.append(new_neuron)
        # for row in self.inputs.iterrows:

    # def calc_random_wieght(self, num_weights):
    #     print("printing random numbers")
    #     return random.uniform(-1.0, 1.0)

    def train_netwrok(self):
        print("random wieghts")
        for row in self.inputs.iterrows():
            print("row")
            for nueron in self.neurons:
                activation = nueron.calculate_activation(row)
                print(activation)



    class Neuron:
        def __init__(self, num_inputs, label):
            self.threshold = 0
            self.label = label
            self.inputs = []
            self.range_max = 1.0
            self.range_min = -1.0
            self.bias = -1
            for i in range(num_inputs):
                self.inputs.append(self.Input(i, random.uniform(self.range_min, self.range_max)))

        def calculate_activation(self, input):
            sum = 0
            for i in range(len(self.inputs)):
                sum += self.inputs[i].weight * input[1][i]
            if sum > self.threshold:
                return 1
            else:
                return 0

        class Input:
            def __init__(self, attribute, weight):
                self.attribute = attribute
                self.weight = weight


the_data_set = Dataset('iris.csv', .3, 42, False)

neural = NeuralNetwork(3, the_data_set.data_train, the_data_set.target_train, the_data_set.data_test)

neural.train_netwrok()
#output = decisionTree.get_highest_entropy(the_data_set.data)