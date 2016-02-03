import numpy as np
from scipy.stats import mode
from sklearn.cross_validation import train_test_split
from scipy.stats import entropy as entropy
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import sys


class Dataset:
    def __init__(self, csv, test_size, random_state, convert_nominal=False):

        #Read in the csv
        self.ds = pd.read_csv(csv, header=None)
        #get the  column length
        ds_num_col = len(self.ds.columns)

        self.data_train = []
        self.data_test = []
        self.target_train = []
        self.target_test = []
        self.std_train = []
        self.std_test = []

        #print("The dataframe before: \n", self.ds)
        #if the user wants numerical data, convert it
        if convert_nominal is True:
            self.convert_numerical_data(self.ds)
        else: #otherwise convert to no
            self.convert_nominal_data(self.ds)
            self.standardize_data()
       # print("The dataframe after: \n", self.ds)

        self.data = self.ds.loc[:, : ds_num_col - 2]
        self.targets = self.ds[ds_num_col - 1]

        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
                self.data, self.targets, test_size=test_size, random_state=random_state)
        self.std_train = [0] * int(len(self.data_train))
        self.std_test = [0] * int(len(self.data_test))

       #print("train data and targets", self.data_train, self.target_train)

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


    def __init__(self, data, targets, inputs):
        self.total_data = data
        self.tree = None
        self.targets = targets
        self.inputs = inputs
        self.classes = np.unique(targets)
        self.nInputs = np.shape(inputs)[0]
        self.closest = np.zeros(self.nInputs)
        self.target_list = np.unique(targets)
        self.attribute_list = []
        for i in range(0,len(self.total_data.columns)):
            print("INDEX@!!!@!@", i)
            self.attribute_list.append(self.Attribute(i))
        print("ATTRIBUTE LIST: ", self.attribute_list)
        print("NUM INPUTS:\n ", self.nInputs)

    class Attribute():
        def __init__(self, name):
            self.name = name
            self.entropy = 0

    def get_highest_entropy(self, data):
        entropies = []
        for i in range(0, len(self.attribute_list)):
            unique_items =  np.unique(data[i])
            print("The unique items: ", unique_items)
            probs = self.calculate_prob(data[i], unique_items)
            my_entropy = entropy(probs,qk=None, base=2)
            self.attribute_list[i].entropy = my_entropy
        print("ATTRIBUTE LIST:")
        for i in self.attribute_list:
            print(i.name, i.entropy)

    def calculate_prob(self, col, unique_items):
        counts = []
        probs = []
        for i in unique_items:
            probs.append(col.value_counts(1)[i])
            counts.append(col.value_counts(0)[i])
        print(counts)
        print(probs)
        return probs

    class AttributeNode():
        def __init__(self, node_col, data_set, is_root_node=False):
            self.node_col = node_col
            self.data_set = data_set
            self.children = []
            self.is_root_node = is_root_node

        def add_child(self, obj):
            self.children.append(obj)

        def print_tree(self, level=1):
            print(("\t" * level), self.node_col)
            if self.children:
                for child in self.children:
                    if isinstance(child, str):

                        print(("\t" * (level + 1)) + child)
                    else:
                        child.print_tree(level+1)

        def isLeaf(self):
            return True;

    class LeafNode():
        def __init__(self, catagory):
            self.catagory = catagory

        def print_tree(self):
            print(self.catagory)
        def isLeaf(self):
            return True;



    class DataPoint():
        def __init__(self, name):
            self.name = name
            self.count = 0
        def increment(self):
            self.count += 1



    def build_tree(self, node, data, targets, attribute_list):
        #lowest_entropy = max(attribute.entropy for attribute in self.attribute_list)
        if(self.attribute_list):
            lowest_entropy_index = min(enumerate(attribute_list), key=lambda x: x[1].entropy)[0]
            print("lowest entropy index: ", lowest_entropy_index)
            if node is None:
                node = self.AttributeNode(lowest_entropy_index, data, True)
            else:
                node.add_child(self.AttributeNode(lowest_entropy_index, data))

            #data_point_counts = self.DataPointCounts(self.target_list)

            for i in np.unique(data[lowest_entropy_index]):
                new_set = data[data[lowest_entropy_index] == i]
                print("new set of: ", i, "\n", new_set.index.values)
                set_indicies = new_set.index.values
                data_points = []
                for target in self.target_list:
                    data_points.append(self.DataPoint(target))
                    #count occurences of each target in the new_set
                for index in set_indicies:
                    for j in data_points:
                        if j.name is targets[index]:
                            j.increment()

                pure_tracker = 0
                for data_point in data_points:
                    if data_point.count > 0:
                        leaf_name = data_point.name
                        pure_tracker += 1
                    print("count for: ", data_point.name, ": ", data_point.count)
                print("TREE: ", node.print_tree())
                if pure_tracker > 1:
                    #make attribute node
                    print("deleting lowest entropy index: ", lowest_entropy_index)
                    copy_list = attribute_list
                    copy_list.remove(copy_list[lowest_entropy_index])
                    self.build_tree(node, new_set, targets, copy_list)
                else:
                    node.add_child(leaf_name)






print("**********************CSV**************************")

my_dataset = Dataset('iris.csv', .3, 3333, True)

decision_tree = DecisionTree(my_dataset.data_train, my_dataset.target_train, my_dataset.data_test)

decision_tree.get_highest_entropy(decision_tree.total_data)
decision_tree.build_tree(decision_tree.tree, decision_tree.total_data, decision_tree.targets, decision_tree.attribute_list)
decision_tree.tree.print_tree()

#closest = nearestneighbor.knn()

#closest = closest.astype(int)

#print("closest: ", closest)

#acc_score = accuracy_score(my_dataset.target_test, closest)

#print("Accuracy of .knn() is: ", acc_score)

