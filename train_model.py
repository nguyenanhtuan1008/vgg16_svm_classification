import numpy as np
import os
from sklearn import svm
from sklearn import metrics
import joblib

def load_data(PATH):
    data = []
    labels = []
    i = 0
    folderList = os.listdir(PATH)
    print("[+] Loading data")
    for folder in folderList:
        fileList = os.listdir(os.path.join(PATH, folder))
        for file in fileList:
            filePath = os.path.join(PATH, folder, file)
            x = np.load(filePath)[0]
            print(x)
            data.append(x)
            if folder == "man":
                label = 0
            else:
                label = 1
            labels.append([label])
            i = i + 1
            print("Loaded", i)

    data = np.array(data,dtype="float")/255.0
    labels = np.array(labels)
    print("Done loading")
    return data, labels


def train_model_SVMLinear(dataTrain, labelTrain, dataTest, labelTest):
    print("[+] Training model")
    clf = svm.SVC(kernel='linear')
    clf.fit(dataTrain, labelTrain)
    print("Done training")
    pd = clf.predict(dataTest)
    print("[+] Testing model")
    print("Testing accuracy: ", metrics.accuracy_score(labelTest, pd))
    joblib.dump(clf, "train_model.pkl")
    print('Model save as "train_model.pkl"')
    return clf

trainPath = "train_x/"
trainData, trainLabels = load_data(trainPath)

testPath = "test_x/"
testData, testLabels = load_data(testPath)

train_model_SVMLinear(trainData, trainLabels, testData, testLabels)
