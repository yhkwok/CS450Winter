from sklearn import datasets
import random
#import iris
iris = datasets.load_iris()

#create the array to work with
theArray = []
for i, iris.data in enumerate(iris.data):
    if i <= 49:
        targets = 0
    elif i <= 99 and i > 49:
        targets = 1
    else:
        targets = 2
    tempArray = [iris.data, targets]
    theArray.append(tempArray)
#shuffle it
random.shuffle(theArray)
#split it
print(theArray)
trainset = theArray[:105] # 70%
testset = theArray[105:] # 30%

class HardCoded(object):
    def __init__(self):
        return
    def train(self, trainset):
        return

    def predict(self, testset):
        predictions = []
        for i, data in enumerate(testset):
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
#print(predictions)
#eva
percentage = classifier.evaluate(predictions, testset)
percentage *= 100
print(percentage)
#print("%.2f" % percentage) + "%"

