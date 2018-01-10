# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:57:36 2017

@author: leejia
"""
import sys
import os
import re
import collections
import math
from stop_words import get_stop_words

#!/usr/bin/python 


if __name__ == '__main__':
    #Reading two command-line arguments, location of training root folder and 
    #test root folder
    global trainrootfolder
    trainrootfolder = sys.argv[1:][0]
    global testrootfolder 
    testrootfolder = sys.argv[1:][1]
    
    #Using Python's stop-words package to get the stop words in English
    stop_words = get_stop_words('english')
                
    #Extracting Vocabulary from the training dataset, ignoring stop words
    countDocuments = 0
    vocabularyCount = 0
    vocabulary = []
    vocabularySet = set()
    for root, dirs, files in os.walk(trainrootfolder):
        for file in files:
            with open(os.path.join(root, file), "r") as auto:
                countDocuments = countDocuments + 1
                for line in auto:
                    if 'Lines:' in line:
                        for line in auto:
                            line = line.lower()
                            words = re.split('\W', line)
                            for word in words:                        
                                if word not in vocabularySet and word not in stop_words:
                                    vocabularySet.add(word)
                                    vocabulary.append(word)
                                    vocabularyCount = vocabularyCount + 1
    
    #Calculating prior of each class
    i = 0
    classPath = []
    prior = []
    for root, dirs, files in os.walk(trainrootfolder):
        if (i>0):
            classPath.append(root)
            priorValue = len(files) / countDocuments;
            prior.append(priorValue)
        i = i + 1
    
    
    w, h = 5, vocabularyCount;
    vocabularyClass = [[0 for x in range(w)] for y in range(h)] 
    countDocumentsClass = [0 for x in range(w)]
    vocabularyCountClass = [0 for x in range(w)]
    countTokensOfTermClass = [0 for x in range(w)]
    condProbabilityOfTermClass = [0 for x in range(w)]
    
    #for each class counting number of each vocabulary tokens
    for i in range(0,5):        
        countDocumentsClass[i] = 0
        vocabularyCountClass[i] = 0
        for root, dirs, files in os.walk(classPath[i]):
            for file in files:
                with open(os.path.join(root, file), "r") as auto:
                    countDocumentsClass[i] = countDocumentsClass[i] + 1
                    for line in auto:
                        if 'Lines:' in line:
                            for line in auto:
                                line = line.lower()
                                words = re.split('\W', line)
                                for word in words:                        
                                    if word not in stop_words:
                                        vocabularyClass[i].append(word)
                                        vocabularyCountClass[i] = vocabularyCountClass[i] + 1
        countTokensOfTermClass[i] = collections.Counter()
        for word in vocabularyClass[i]:
            countTokensOfTermClass[i][word] += 1
        
        #calculating conditional probability of each term in the vocabulary
        countTokensOfTermValue = 0
        condProbabilityOfTermValue = 0
        condProbabilityOfTermClass[i] = collections.Counter()
        for word in vocabulary:
            countTokensOfTermValue = countTokensOfTermClass[i][word]
            condProbabilityOfTermValue = (countTokensOfTermValue + 1)/(vocabularyCountClass[i] + vocabularyCount)
            condProbabilityOfTermClass[i][word] = condProbabilityOfTermValue
        
        
    #Testing the model using test dataset
    countTestDocuments = 0
    countDocumentsWronglyClassified = [0 for x in range(w)]
    countDocumentsWronglyClassifiedTotal = 0
    #extracting tokens from document
    for root, dirs, files in os.walk(testrootfolder):
        for file in files:
            with open(os.path.join(root, file), "r") as auto:   
                countTestDocuments = countTestDocuments + 1
                vocabularyCountTest = 0
                vocabularyTest = []
                vocabularySetTest = set()
                for line in auto:
                    if 'Lines:' in line:
                        for line in auto:
                            line = line.lower()
                            words = re.split('\W', line)
                            for word in words:                        
                                if word not in vocabularySetTest and word not in stop_words:
                                    vocabularySetTest.add(word)
                                    vocabularyTest.append(word)
                                    vocabularyCountTest = vocabularyCountTest + 1
                scoreClass = [0 for x in range(5)]
                maxScore = 0
                maxScoreClass = 0
                condProbabilityOfTermValue = 0
                #calculating score and returning maximum score class
                for i in range(0,5):
                    scoreClass[i] = math.log(prior[i])
                    for word in vocabularyTest:
                        #handling new words
                        if condProbabilityOfTermClass[i][word] == 0:
                            condProbabilityOfTermValue = (0 + 1)/(vocabularyCountClass[i] + vocabularyCount)
                            scoreClass[i] = scoreClass[i] + math.log(condProbabilityOfTermValue)
                        else:
                            scoreClass[i] = scoreClass[i] + math.log(condProbabilityOfTermClass[i][word])
                    if i == 0:
                        maxScore = scoreClass[i]
                        maxScoreClass = i
                    elif scoreClass[i] > maxScore:
                        maxScore = scoreClass[i]
                        maxScoreClass = i
                
                #finding out wrongly classified documents
                if ("rec.autos" in auto.name and maxScoreClass != 0):
                    countDocumentsWronglyClassified[0] = countDocumentsWronglyClassified[0] + 1
                elif ("rec.sport.hockey" in auto.name and maxScoreClass != 1):
                    countDocumentsWronglyClassified[1] = countDocumentsWronglyClassified[1] + 1
                elif ("sci.med" in auto.name and maxScoreClass != 2):
                    countDocumentsWronglyClassified[2] = countDocumentsWronglyClassified[2] + 1
                elif ("sci.space" in auto.name and maxScoreClass != 3):
                    countDocumentsWronglyClassified[3] = countDocumentsWronglyClassified[3] + 1
                elif ("soc.religion.christian" in auto.name and maxScoreClass != 4):
                    countDocumentsWronglyClassified[4] = countDocumentsWronglyClassified[4] + 1
       
    i = 0;
    #calculating accuracy separately for each test sub folders
    for root, dirs, files in os.walk(testrootfolder):
        if (i>0):
            print("")
            print(root)
            print("=====================================")
            print("Number of Test Documents : ",len(files));
            print("Number of Test Documents classified correctly : ",len(files) - countDocumentsWronglyClassified[i-1])
            accuracy = ((len(files)-countDocumentsWronglyClassified[i-1])/len(files))*100
            print("Accuracy : ",accuracy,"%")           
            countDocumentsWronglyClassifiedTotal = countDocumentsWronglyClassifiedTotal + countDocumentsWronglyClassified[i-1]
        i = i+1
        
    #calculating accuracy of the model, in general
    print("")
    print("Total Number of Test Documents : ",countTestDocuments);
    print("Total Number of Test Documents classified correctly : ",countTestDocuments - countDocumentsWronglyClassifiedTotal)
    accuracy = ((countTestDocuments-countDocumentsWronglyClassifiedTotal)/countTestDocuments)*100
    print("Accuracy : ",accuracy,"%")      
    
