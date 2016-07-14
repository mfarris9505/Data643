# -*- coding: utf-8 -*-
"""
Created on Sun Jul 03 12:30:45 2016

@author: Matts42
"""

import pandas as pd 
import numpy as np
import csv
import math
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#Data file containing ratings data
f = open("data/u.data")
text = csv.reader(f, delimiter = "\t")
ratings = list(text)

#Data containing Item Data (Content)
f = open("data/u.item")
text = csv.reader(f, delimiter = "|")
item = list(text)


ratings_pd = pd.DataFrame(ratings, columns = ["user_id","movie_id", "rating","timestamp"])
ratings_pd = ratings_pd.drop(["timestamp"], 1)

movies = ratings_pd.pivot(index = "user_id", columns = "movie_id", values = "rating")

movies = movies.apply(pd.to_numeric, errors='coerce')
movies_int = np.array(movies)
np.nan_to_num(movies_int)

# Borrowed some code here from Padebetttu et. al. -DATA643- Project 2
user_mean = movies.T.mean(skipna=True)
item_mean = movies.mean(skipna=True)
movies = movies.apply(lambda x: x.fillna(x.mean(skipna = True)),axis=1)
movies_mean = movies.apply(lambda x: x-x.mean(), axis = 1).round(5)

#Using SVD to find 
movies_np = np.array(movies_mean)
movies_np = movies_np.astype(float)


def svd_red(movies_mean,n):
    U1, s1, V1 = np.linalg.svd(movies_mean, full_matrices=True)
    k = np.zeros((len(s1),len(s1)),float)
    np.fill_diagonal(k,s1)
    k = k[:n,:n]
    k = np.sqrt(k)
    U2 = U1[:,:n]
    V2 =V1[:,:n].T
    Uk = np.dot(U2,k.T)
    Vk = np.dot(k,V2)
    
    
    R_red = np.dot(Uk,Vk)
    return R_red, Uk, Vk

def pred_item(movies_mean,n):
    R_red, Uk, Vk = svd_red(movies_mean,n)
    item_similarity = 1 - pairwise_distances(Vk.T, metric='cosine')
    pred = R_red.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])  
    final_pred = pred.T
    final_pred += user_mean
    final_pred = final_pred.T
    return final_pred

def als_red(movies, n):
    
MAE =[]    
RMSE = []

k = [1,2,3,4,5,10,20,30,40,50,100,200,300,400,500]

for i in k:
    pred = pred_item(movies_mean, i)
    y = mean_squared_error(movies_int[np.where(movies_int >0)], pred[np.where(movies_int >0)])   
    RMSE.append(math.sqrt(y))
    
    
