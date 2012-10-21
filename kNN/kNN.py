from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, .1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in xrange(k):
        voteILabel = labels[sortedDistIndices[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2Matrix(fname, noOfParams):
    fr = open(fname)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines, noOfParams))
    classLabelVector = []
    fr = open(fname, 'r')
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:noOfParams]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, labels = file2Matrix('datingTestSet.txt', 3)
    normMat, ranges, minValue = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in xrange(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], labels[numTestVecs:m], 3)
        print "The classifier came back with: %s, the real answer is: %s" % (classifierResult, labels[i])
        if(classifierResult != labels[i]):
            errorCount += 1.0
    print "The total error rate is : %f" % (errorCount / float(numTestVecs))


def classifyPerson():
    percentTats = float(raw_input("Time spent playing?"))
    ffMIiles = float(raw_input("FF miles per year?"))
    iceCream = float(raw_input("Liters of ice cream consumed per year?"))
    datingDataMat, labels = file2Matrix('datingTestSet.txt', 3)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMIiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, labels, 3)
    print 'You will most probably: %s' % classifierResult
