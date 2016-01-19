# collaborated with Jonathan Kwok
import random
import math
import csv
from sklearn import datasets
import numpy as np
from numpy import array

# create a new array that will store the our data and targets
myArray = []
numArgs = 0
var = input("Which data set would you like to use? \n A. Iris \n B. Cars (Note: Cars will take awhile to run) \n ")

def convert (carArray):
    numArgs = 6
    for i in range(len(carArray)):
        j = 0
        while j < 6:
            if (carArray[i][0][j] == 'low' or carArray[i][0][j] == 'small'):
                carArray[i][0][j] = 1
            elif (carArray[i][0][j] == 'med' or carArray[i][0][j] == '2'):
                carArray[i][0][j] = 2
            elif(carArray[i][0][j] == 'high' or carArray[i][0][j] == '3' or carArray[i][0][j] == 'big'):
                carArray[i][0][j] = 3
            elif (carArray[i][0][j] == 'vhigh' or carArray[i][0][j] == '4'):
                carArray[i][0][j] = 4
            elif (carArray[i][0][j] == '5more'):
                carArray[i][0][j] = 5
            elif (carArray[i][0][j] == 'more'):
                carArray[i][0][j] = 6
            else:
                carArray[i][0][j] = 1
            j+=1

    for i in range(len(carArray)):
        if (carArray[i][1] == 'unacc'):
            carArray[i][1] = 1
        elif (carArray[i][1] == 'acc'):
            carArray[i][1] = 2
        elif (carArray[i][1] == 'good'):
            carArray[i][1] = 3
        else:
            carArray[i][1] = 4

    return carArray


if (var == 'A' or var == 'a'):
    iris = datasets.load_iris()
    numArgs = 4
    #Fill myArray with iris data and target.
    for i, iris.data in enumerate(iris.data):
        if i <= 49:
            targets = 0
        elif i <= 99 and i > 49:
            targets = 1
        else:
            targets = 2
        tempArray = [iris.data, targets] #assign targets and data to an array with everything in place
        myArray.append(tempArray)

if (var == 'B' or var == 'b'):
    carArray = []
    dataset = []
    f = open('car.csv')
    csv_f = csv.reader(f)
    for row in csv_f:
        i = [row[0],row[1],row[2],row[3],row[4],row[5]]
        dataset.append(i)
        tempArray = [i, row[6]]
        carArray.append(tempArray)
    convert(carArray)
    myArray = carArray

#Randomize our array
random.shuffle(myArray)

#separate training set and test set
sizeTrainingSet = len(myArray) * .7
trainingSet = myArray[:int(sizeTrainingSet)] #70%
testSet = myArray[int(sizeTrainingSet):]  #30%

class HardCoded:
    def train(self, trainingSet):
        return 0

#********************************************************************
# Predict method to return an array of target predictions
#
#********************************************************************
    def predict(self, testSet, trainingSet):
        kCount = 4
        predictions = []
        top = []
        for i, data in enumerate(testSet):
            distanceArray= []
            nearestNeighbors = []
            for j, data in enumerate(trainingSet):
                k = 0
                tempNum = 0.0
                # Distance algorithm
                while (k < numArgs):
                    tempNum += math.pow(testSet[i][0][k] - trainingSet[j][0][k], 2)
                    k+=1
                t = math.sqrt(tempNum)
                tempArray = [t, trainingSet[j][1]]
                # make an array with distances and targets
                distanceArray.append(tempArray) # this is the important array of distances and targets. we need to find the smallest distances.
                # sort
                distanceArray.sort(key=lambda  ele : ele[0])
            count = 0
            # grab the nearest neighbors
            while (count < kCount):
                    tempArray2 = distanceArray[count]
                    nearestNeighbors.append(tempArray2)
                    count +=1

            nearestTargets = [x[1] for x in nearestNeighbors]
            from collections import Counter
            word_counts = Counter(nearestTargets)
            top = word_counts.most_common(1)
            target = top[0][0] #target will equal 0, 1, or 2
            predictions.append(target)
        return predictions

#********************************************************************
# Evaluate method to return the percentage of correctness of our predictions
#
#********************************************************************
    def evauluate(self, predictions, testSet):
        correctCount = 0.0
        totalCount = 0.0
        for i, data in enumerate(testSet):
            totalCount +=1
            if predictions[i] == data[1]:
                correctCount +=1
        percentage = correctCount / totalCount
        return percentage

train = HardCoded()

train.train(trainingSet)
predictions = train.predict(testSet, trainingSet)# an array of 0's
percentage =  train.evauluate(predictions, testSet)
finalAnswer = percentage * 100
print("success rate:")
print ("%.2f" % finalAnswer + "%")