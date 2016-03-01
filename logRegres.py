# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 14:53:04 2016

@author: nju-hyhb
"""
import __future__
from numpy import *

def collectData(filename):
    fr=open(filename)
    returnX=[];returnXstr=[]
    returnY=[]
    for line in fr.readlines():
        lineStr=line.strip().split('\t')
        length=len(lineStr)
        returnXstr.append((lineStr[0:(length-2)]))
        returnY.append(float(lineStr[-1]))
    for line in returnXstr:
        for str in line:
            returnX.append(float(str))
    returnX=reshape(returnX,(-1,length-2))            
    return returnX,returnY
    
def sigmod(X):
    return 1.0/(1+exp(-X))
    
def logRegTraining(X,Y,theta_in,num):
    rows,cols=shape(X)
    minimize=sqrt(sum((Y-dot(X,theta_in))**2))
    if num<rows:
        for i in range(rows):
            h=sigmod(sum(X[i]*theta_in))
            alpha=0.01/(i+1)
            for j in range(cols):
                theta[j]=theta_in[j]+alpha*(Y[i]-h)*X[i,j]
            minimize_new=sqrt(sum((Y-dot(X,theta))**2))
            if minimize_new<minimize:
                minimize=minimize_new
                theta_in=theta
    else: print 'The training times is too large.'    
    return theta_in

def classify0(Data,theta):
    rows,cols=shape(Data)
    labels=zeros(rows)
    for i in range(rows):
        y=sigmod(sum(Data[i]*theta))
        if y>0.5:
            labels[i]=1
    return labels

def main():
    filename=('horseColicTraining.txt')
    X,Y=collectData(filename)
    rows,cols=shape(X)
    theta=rand(cols)
    times=200
    theta=logRegTraining(X,Y,theta,times)
    
    testfile=('horseColicTest.txt')
    testData,testLabel=collectData(testfile)
    
    classifyLabels=classify0(testData,theta)
    
    errorcount=0
    for i in range(size(testLabel)):
        if testLabel[i]!=classifyLabels[i]:
            errorcount+=1
    print 'The error rate is:',float(errorcount)/size(testLabel)

if __name__=='__main__':
    main()



