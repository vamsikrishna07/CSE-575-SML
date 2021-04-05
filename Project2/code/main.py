# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 00:36:52 2021

@author: Vamsi
"""

import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def plot(x,y,clr,xlabel,ylabel,title,isScatter):
    if isScatter: plt.scatter(x,y,color=clr,marker='s')
    else: plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def strategy1(data,clusters):
    list1, list2 = [],[]
    for each in clusters: 
        initialCentroids = (data.sample(n=each))
        difference, n = 1, 0   
        list1,list2 = kMeans(difference,data,initialCentroids,each,n,list1,list2)
    plot(list1,list2,'red',"Cluster Size -k","Loss Computed by Objective function","KMeans plot (Cluster Size VS Loss Function)-Strategy 1",False)

def strategy2(data,clusters):
    list1, list2 = [],[]
    for each in clusters: 
        initialCentroids = getInitialCentroids(data,each)
        difference, n = 1, 0   
        list1, list2 = kMeans(difference,data,initialCentroids,each,n,list1, list2)
    plot(list1,list2,'red',"Cluster Size -k","Loss Computed by Objective function","KMeans plot (Cluster Size VS Loss Function)-Strategy 2",False)    

def getInitialCentroids(data, each):
    firstCentroid = (data.sample(n=1))
    centroidList = pd.DataFrame()
    centroidList = centroidList.append(firstCentroid,ignore_index=True)
    for every in range(each-1):
        max_distance = 0.0
        for id1,row_k in data.iterrows():
            total_distance = 0.0
            for id2,row_every in centroidList.iterrows():
                distance = np.sqrt((row_k['x']-row_every['x'])**2+(row_k['y']-row_every['y'])**2) 
                total_distance += distance
            distanceAverage = total_distance/len(centroidList)
            if  distanceAverage>max_distance:
                max_distance = distanceAverage
                d = id1
        centroidList = centroidList.append(data.iloc[d,:],ignore_index = True)
    return centroidList

def kMeans(difference,data,initialCentroids,each,n,list1, list2):
    while(difference!=0):
        dataX,i,cluster = data,0,[]
        data,i = getEuclideanDistance(initialCentroids,dataX,data,i)
        cluster = groupClusters(data,each,cluster)
        data["Cluster"] = cluster
        new_Centroids = data.groupby(["Cluster"]).mean()[['x','y']]
        list3,difference,n = compareCentroids(new_Centroids,initialCentroids,data,difference,n,each)
        initialCentroids = new_Centroids
    colors=['red','yellow','green','violet','pink','chocolate','cyan','orange','wheat','grey','c']
    multiColorPlot(initialCentroids,data,colors)
    plt.show()
    list2.append(sum(list3))
    list1.append(each)
    return list1,list2  
      
def getEuclideanDistance(initialCentroids,dataX,data,i):
    for index1,row1 in initialCentroids.iterrows():
            euclideanDistance = []
            for index2,row2 in dataX.iterrows():
                d1 = (row1['x']-row2['x'])**2
                d2 = (row1['y']-row2['y'])**2
                d = np.sqrt(d1+d2)
                euclideanDistance.append(d)
            data[i] = euclideanDistance 
            i += 1
    return data,i

def multiColorPlot(initialCentroids,X,colors):
    for index1,row1 in initialCentroids.iterrows():
            xlist, ylist = [],[]
            for index2,row2 in X.iterrows():
                if row2["Cluster"]==index1:
                    xlist.append(row2['x'])
                    ylist.append(row2['y'])
            plt.scatter(xlist,ylist,c=colors[index1],marker='s')
            plt.scatter(row1['x'],row1['y'],c='black',marker='^')
            plt.xlabel('X')
            plt.ylabel('Y')
            
def compareCentroids(newCentroids,initialCentroids,data,difference,n,each):
    for index1,row1 in newCentroids.iterrows():
            list1 = []
            for index2,row2 in data.iterrows():
                if row2["Cluster"]==index1:
                    d1=(row1['x']-row2['x'])**2
                    d2=(row1['y']-row2['y'])**2
                    list1.insert(each,d1+d2)
    if n == 0: difference,n = 1,n+1
    else: difference = (newCentroids['x'] - initialCentroids['x']).sum() + (newCentroids['y'] - initialCentroids['y']).sum()
    return list1,difference,n
    
def groupClusters(data,each,clusters):
    for index,row in data.iterrows():
        minDistance,position = row[0],0
        for i in range(each):
            if row[i] < minDistance:
                minDistance = row[i]
                position=i
        clusters.append(position)
    return clusters

def main():
    file = scipy.io.loadmat('AllSamples.mat',squeeze_me=True)
    data = pd.DataFrame(file['AllSamples'])
    print(data.size)
    data.rename(columns = {0: "x", 1:"y"}, inplace = True)
    clusters = []
    for x in range(2,11): clusters.append(x)
    plot(data['x'],data['y'],'red','X','Y','InitialPlot','True')
    for i in range(1,3):
        strategy1(data,clusters)
        strategy2(data,clusters)

if __name__ == '__main__':
    main()