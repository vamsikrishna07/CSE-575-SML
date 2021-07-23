"""
@author: Vamsi
"""
import scipy.io as sp
import pandas as pd
import numpy as np
import math
#FUNCTION TO GENERATE FEATURES(MEAN AND STANDARD DEVTATION)
def GenerateFeatures(d): 
    res = []
    d.apply(lambda x: res.append([pd.Series.mean(x),pd.Series.std(x)]),axis=1)
    return pd.DataFrame(res)
#FUNCTION TO FIND THE ACCURACY
def Accuracy(testY,predY):
    tshirtCount, trouserCount, isTshirt, isTrouser = len(testY)//2,len(testY)//2,0,0
    for i in range(len(testY)):
        if testY.loc[i][0]==predY.loc[i][0]:
            if testY.loc[i][0]==0.0: isTshirt += 1
            else: isTrouser += 1
    return (isTshirt/tshirtCount),(isTrouser/trouserCount), ((isTshirt+isTrouser)/(tshirtCount+trouserCount))
#FUNCTION TO IMPLEMENT NAIVE BAYES CLASSIFICATION
def NaiveBayes(trainset, testX, testY): 
    def gaussianProbability(a,mean,std): #FUNCTION TO FIND GAUSSIAN PROBABILITY
        x = 1/(math.sqrt(2*math.pi)*std)
        y = math.exp(-math.pow(a-mean,2)/(2*math.pow(std,2)))
        return x*y
    #FUNCTION TO FIND POSTERIOR PROBABILITY
    def postProbability(testrow,mean,std): 
        return gaussianProbability(testrow[0],mean[0],std[0])*gaussianProbability(testrow[1],mean[1],std[1])
    #FUNCTION TO FIND THE PREDICTED VALUE FOR TEST DATA
    def naiveBayesPrediction(testX,trainTshirtMean,trainTshirtStd,trainTrouserMean,trainTrouserStd,tShirtProbability,trouserProbability):
        predY = []
        for i,testrow in testX.iterrows():
            tshirt_prob = tShirtProbability*postProbability(testrow,trainTshirtMean,trainTshirtStd)
            trouser_prob = trouserProbability*postProbability(testrow,trainTrouserMean,trainTrouserStd)
            if tshirt_prob > trouser_prob: predY.append(0.0)
            else: predY.append(1.0)
        return pd.DataFrame(predY)
    #DIVIDING THE TRAIN DATA BASED ON THEIR CLASS 
    trainTshirt,trainTrouser = trainset.iloc[:len(trainset)//2,:],trainset.iloc[len(trainset)//2:,:]
    #FINDING TOTAL MEAN AND STANDARD DEVIATION FOR EACH CLASS
    trainTshirtMean,trainTshirtStd= trainTshirt.mean()[:-1],trainTshirt.std()[:-1]
    trainTrouserMean,trainTrouserStd= trainTrouser.mean()[:-1],trainTrouser.std()[:-1]
    #AS WE HAVE EQUAL NUMBER OF IMAGES IN BOTH CLASSES, PRIOR PROBABILITY IS 0.5
    tShirtProbability, trouserProbability = 0.5, 0.5
    #PREDICTING VALUES FOR TEST DATA
    predY = naiveBayesPrediction(testX,trainTshirtMean,trainTshirtStd,trainTrouserMean,trainTrouserStd,tShirtProbability,trouserProbability)
    #FINDING AND PRINTING ACCURACY
    tshirtAccuracy, trouserAccuracy, naiveBayesAccuracy = Accuracy(testY, predY)
    print('Accuracy for Naive Bayes Classification')
    print('for T-Shirt',str(tshirtAccuracy*100)+'%')
    print('for Trouser',str(trouserAccuracy*100)+'%')
    print('Overall Accuarcy',str(naiveBayesAccuracy*100)+'%')
#FUNCTION TO IMPLEMENT LOGISTIC REGRESSION
def LogisticRegression(trainset, testX, testY): 
    #INITALIZING WEIGHTS, LEARNING RATE AND NUMBER OF EPOCS
    print('\n\n\n\n\nlogistic Regression')
    w0, w1, w2 = 0,0,0
    epochs, learning_rate = 30, 0.3
    #FUNCTION TO FIND THE SIGMOID FUNCTION
    def sigma(mean,std,w0,w1,w2):
        yh =  w0 + w1*mean + w2*std
        den = math.exp(-1*yh)
        return 1.0/(1.0 + den)
    #FUNCTION FOR GRADIENT ASCENT
    def updateWeight(w,pred,x,error):
        return w + (learning_rate*error*pred*(1.0-pred)*x)
    #AS THE DATA IS ORDERED,WE MUST NEED TO SHUFFLE FOR GETTING ACCURATE RESULTS
    np.random.shuffle(trainset.values)
    #ITEARTING THROUGH EACH EPOCH AND UPDATING THE WEIGHTS ACCORDING TO GRADIENT ASCENT
    for e in range(epochs):
        print('epoch - ',e)
        for i in range(len(trainset)):
            mean, std, label = trainset.loc[i][0], trainset.loc[i][1], trainset.loc[i][2]
            pred = sigma(mean,std,w0,w1,w2)
            error = label-pred
            w0 = updateWeight(w0,pred,1,error)
            w1 = updateWeight(w1,pred,mean,error)
            w2 = updateWeight(w2,pred,std,error)
    #INITALIZING A LIST TO STORE THE PREDICTED VALUES 
    y = []
    #PREDICTING THE CLASS LABEL FOR TEST DATA AND STORING IN A DATAFRAME
    for i in range(len(testX)):
        mean, std = testX.loc[i][0], testX.loc[i][1]
        pred = sigma(mean,std,w0,w1,w2)
        if pred>0.5: y.append(1.0)
        else: y.append(0.0)
    predY = pd.DataFrame(y)
    #FINDING THE ACCURACY
    tshirtAccuracy, trouserAccuracy, LRAccuracy = Accuracy(testY, predY)
    print('Accuracy for Logistic Regression')
    print('for T-Shirt',str(tshirtAccuracy*100)+'%')
    print('for Trouser',str(trouserAccuracy*100)+'%')
    print('Overall Accuarcy',str(LRAccuracy*100)+'%')
#MAIN FUNCTION 
if __name__ == "__main__":
    data = sp.loadmat(r'fashion_mnist.mat')
    trainX = pd.DataFrame(data['trX'])
    trainY = pd.DataFrame(data['trY']).transpose()
    testX = pd.DataFrame(data['tsX'])
    testY = pd.DataFrame(data['tsY']).transpose()
    train_features = GenerateFeatures(trainX)
    test_features = GenerateFeatures(testX)
    trainset = pd.concat([train_features, trainY.rename(columns={0:2})], axis = 1)
    NaiveBayes(trainset, test_features, testY)
    LogisticRegression(trainset, test_features, testY)