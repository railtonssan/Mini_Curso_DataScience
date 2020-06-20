#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:24:41 2020

@author: railtonsabtos
"""

import numpy as np


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
        k_neighbors = list(k_neighbors[:,1])
        
       
        #majoritaty votation
        classe = votation(k_neighbors)
        
        classes_labels.append(classe)
    return classes_labels


    