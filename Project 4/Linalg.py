# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:40:52 2016

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


def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)

Q = movies_np

W = Q>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)


lambda_ = 0.1
n_factors = 10
m, n = Q.shape
n_iterations = 20


X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)

weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)
#print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))

weighted_Q_hat = np.round(weighted_Q_hat, decimals = 3)
weighted_errors = np.round(weighted_errors, decimals = 3)

np.savetxt("Q.csv", weighted_Q_hat, delimiter=",")
np.savetxt("Q_err.csv", weighted_errors, delimiter=",")