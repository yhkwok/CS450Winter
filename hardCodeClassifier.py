from sklearn import datasets
import random
#import iris
iris = datasets.load_iris()

#create the array to work with
theArray = []
for i, data in enumerate(iris.data):
    targets = i/50
    tempArray = [data, targets]
    theArray.append(tempArray)
#shuffle it
random.shuffle(theArray)
#split it
trainset = theArray[:105] # 70%
testset = theArray[105:] # 30%

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

class HardCoded(object):
    def __init__(self):
        return
    def train(self, trainset):
        return

    def predict(self, testset):
        predictions = []
        for i, data in enumerate(testset)
            predictions.append(0)
        return predictions
    def evaluate(self, predictions, testset):
        correctCount = 0.0
        totalCount = 0.0
        for i, data in enumerate(testset):
            totalCount += 1
            if predictions[i] == data[1]:
                correctCount += 1
        percentage = correctCount / totalCount
        return percentage

#create classifier
classifier = HardCoded()
#train it
classifier.train(trainset)
#make prediction
predictions = classifier.predict(testset)
#eva
percentage = classifier.evaluate(predictions, testset)
print("%.2f" % (percentage * 100)) + "%"

