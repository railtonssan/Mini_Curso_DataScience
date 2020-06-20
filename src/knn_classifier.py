#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:24:41 2020

@author: railtonsabtos
"""

import pandas as pd
import numpy as np

data = [[3.9, 1], [2.6, 2], [3.7, 1], [8.9, 2], [10.5,1], [11.6,2]]
data = pd.DataFrame(data, columns=['values', 'labels'])
data


def accuracy(y_test, predict): #predict -> resultado do modelo knn
    size = len(y_test)
    count_true = 0
    for i in range(size):
        if y_test[i] == predict[i]:
            count_true =+ 1
    return (count_true/size)*100

def probability(structure, classe):
    list_majoritary_distances = []
    for i in range(len(structure)):
        if classe == structure[i,1]:
            list_majoritary_distances.append(structure[i,0])
    return np.mean(list_majoritary_distances)
            
        
def votation(list_classes):
    classes_uniques = np.unique(list_classes)
    new_list = []
    for cl in classes_uniques:
        new_list.append([list_classes.count(cl), cl])
        
    votation = max(new_list)
    return votation[1]   #irar retornar a classe com maior repeticoes

    #se k não for definido ele sempre será 1
def KNeighborsClassifier(x_train, y_train, x_test, y_test, k = 1):
    
    # validando valor de k para não ocorrer erro de escolha
    if (k % 2) == 0:
        k= k - 1
       
    elif (k <=0 ):
        print('Não é possivel excutar o algoritimo com valores zeros ou menores do zero')
        print('O valor de k será setado para o valor 1')
        k=1
    
    classes_labels =[]
    classes_prob = []
    for i in range(len(x_test)):
       
        list_distance_individual_instance = []
        
        for j in range(len(x_train)):
            
            #calc euclidean distance
            euclidean_distance = np.sqrt(np.sum(np.power(x_test[i]- x_train[j], 2)))
            list_distance_individual_instance.append([euclidean_distance, y_train[j]])
        
        #ordering distance values   
        list_distance_individual_instance.sort()
        
        #defining k neighbors
        k_neighbors = list_distance_individual_instance[0:k]
        k_neighbors = np.array(k_neighbors)
        print("Struture: ", k_neighbors)
        k_neighbors_class = list(k_neighbors[:,1]) # posição 1 é a label
      
        #majoritaty votation
        classe = votation(k_neighbors_class)
        classes_labels.append(classe)
        
        #majoritary probability
        prob = probability(k_neighbors, classe)
        classes_prob.append(prob)
        
    return classes_labels, classes_prob

x_train = data['values'][0:4].values
y_train = data['labels'][0:4].values
x_test  = data['values'][4:].values
y_test  = data['labels'][4:].values
predict, prob = KNeighborsClassifier(x_train, y_train, x_test, y_test, 4)
acc = accuracy(y_test, predict)
print("Preditos: ", predict)
print("Probabilidade: ", prob)
print("Real: ", y_test)
print("Acuracia: ", acc)