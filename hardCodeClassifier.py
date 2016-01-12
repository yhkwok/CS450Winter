from sklearn import datasets
import random
#import iris
iris = datasets.load_iris()
#shuffle it
random.shuffle(iris.data)

print (len(iris.data))
#split
train_data = iris.data[(len(iris.data)*70)/100:]
test_data = iris.data[:(len(iris.data)*30)/100]

# Show the data (the attributes of each instance)
print ("Data: ")
print(iris.data)

# Show the target values (in numeric format) of each instance
print ("Target value: ")
print(iris.target)

# Show the actual target names that correspond to each number
print ("Actual Target Names: ")
print(iris.target_names)

#whole iris
print ("Whole iris: ")
print(iris)

def hardCode(iris):
    return "setosa"

print (hardCode(iris))

print ("Test set: ")
print (test_data)
print ("Train set: ")
print (train_data)

