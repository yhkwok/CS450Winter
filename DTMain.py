import random
import math
import csv
from sklearn import datasets
import pandas as pd
import numpy as np
from scipy.stats import  entropy


from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

class Dataset:
    def __init__(self, csv, test_size, random_state, convert_nominal=False):
        self.ds = pd.read_csv(csv, header=None)
        ds_column_len = len(self.ds.columns)
        print("The dataframe before: \n", self.ds)
        if convert_nominal is True:
            self.convert_numerical_data(self.ds)
        else:
            self.convert_nominal_data(self.ds)
            self.standardize_data()
        print("The dataframe after: \n", self.ds)
        self.data = self.ds.loc[:, : ds_column_len - 2]
        self.targets = self.ds[ds_column_len - 1]
        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.std_train = []
        self.std_test = []

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                self.data, self.targets, test_size=test_size, random_state=random_state)
        self.std_train = [0] * int(len(self.data_train))
        self.std_test = [0] * int(len(self.data_test))

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



    # This function uses a zscore to standardize the data
    def standardize_data(self):
        # fill std_train with standardized values
        self.std_train = (self.data_train - self.data_train.mean()) / self.data_train.std()
        # fill std_test with standardized values
        self.std_test = (self.data_test - self.data_train.mean()) / self.data_train.std()

class DecisionTree:
    def __init__(self, k, data, targets, inputs):
        self.k = k
        self.data = data
        self.targets = targets
        self.classes = np.unique(targets)
        self.inputs = inputs
        self.nInputs = np.shape(inputs)[0]
        self.closest = np.zeros(self.nInputs)
        self.target_list = np.unique(targets)
        self.attribute_list = list(range(0,len(self.data.columns)))
        print("NUM INPUTS: ", self.nInputs)

    class Attribute():
        def __init__(self, name):
            self.name = name
            self.entropy = 0


        
    def getHiEntropy(self, data):
        entropies = []
        for i in range(0, len(data.columns)):
            unique_items = np.unique(data[i])
            print("Unique Items: ", unique_items)
            probs = self.calculate_prob(data[i], unique_items)
            entropies.append(entropy(probs, qk=None, base = 2))
        print("all entropies", entropies)
        print("highest entropy", np.argmax(entropies))

    def calculate_prob(self, col, unique_items):
        counts = []
        probs = []
        for i in unique_items:
            probs.append(col.value_counts(1)[i])
            counts.append(col.value_counts(0)[i])

        print(probs)
        print(counts)
        return probs

    def buildTree(self, ):


ds = Dataset("iris.csv", .3, 42, True)
print (ds)