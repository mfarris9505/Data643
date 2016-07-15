# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 20:40:52 2016
@author: Matts42
"""
import pandas as pd 
import numpy as np
import csv

#Data file containing ratings data
f = open("data/u.data")
text = csv.reader(f, delimiter = "\t")
ratings = list(text)

#Data containing Item Data (Content)
f = open("data/u.item")
text = csv.reader(f, delimiter = "|")
item = list(text)

#Data containing User-Content Data 
f = open("data/u.user")
text = csv.reader(f, delimiter = "|")
user = list(text)

#Data containing job names
f = open("data/u.occupation")
text = csv.reader(f, delimiter = "\t")
jobs = list(text)

#Extracting Initial Data
user_pd = pd.DataFrame(user, columns = ["user_id","age","gender","occupation","zip"])
user_pd = user_pd.set_index("user_id")
items_pd = pd.DataFrame(item, columns = ["movie_id","title","release","video_release","IMDB",
                                         "unknown","Action","Adventure","Animation","Child",
                                         "Comedy","Crime","Doc","Drama","Fantasy","Film-Noir",
                                         "Horror","Musical","Mystery","Romance","Sci-Fi", 
                                         "Thriller","War", "Western"])

items_pd = items_pd.drop(["movie_id","release","video_release","IMDB"], 1)
ratings_pd = pd.DataFrame(ratings, columns = ["user_id","movie_id", "rating","timestamp"])
ratings_pd = ratings_pd.drop(["timestamp"], 1)
movies = ratings_pd.pivot(index = "user_id", columns = "movie_id", values = "rating")

#Making the User Dataset Matrix friendly
map_gender = {"M":0,"F":1}
map_jobs = {}
Age_range = {"0-18":0, "18-25":1,"25-40":2,"40-60":3, "65-80":4}

#Creathing Jobs Dict
i = 0 
for job in jobs:
    map_jobs[job[0]] = i
    i = i + 1

def Age_Range(x):
    if x in range(19):
        x = 0
        return x
    elif x in range(18,26):
        x = 1
        return x
    elif x in range(25,41):
        x = 2
        return x
    elif x in range(40,66):
        x = 3
        return x
    elif x in range(65,81):
        x = 4
        return x

user_pd['age'] = user_pd['age'].astype(int)
user_pd['age'] = user_pd['age'].map(Age_Range)
user_pd = user_pd.replace({"gender": map_gender,'occupation': map_jobs})

#Creating Dummy col for Jobs
dummies = pd.get_dummies(user_pd['occupation'], prefix='job', prefix_sep='_')
col_names_dummies = dummies.columns.values

for i,value in enumerate(col_names_dummies):
    user_pd[value] = dummies.iloc[:,i]

#Creating Dummy col for age
dummies = pd.get_dummies(user_pd['age'], prefix='age', prefix_sep='_')
col_names_dummies = dummies.columns.values

for i,value in enumerate(col_names_dummies):
    user_pd[value] = dummies.iloc[:,i]

user_pd = user_pd.drop(["age","occupation","zip"], 1)
movies = movies.apply(pd.to_numeric, errors='coerce')
movies_int = np.array(movies)
np.nan_to_num(movies_int)

# Borrowed some code here from Padebetttu et. al. -DATA643- Project 2
user_mean = movies.T.mean(skipna=True)
item_mean = movies.mean(skipna=True)
movies = movies.apply(lambda x: x.fillna(x.mean(skipna = True)),axis=1)
movies_mean = movies.apply(lambda x: x-x.mean(), axis = 1).round(5)

#Finalizing our datasets
movies_np = np.array(movies_mean)
movies_np = movies_np.astype(float)

#items dataset 
items = items_pd.set_index(["title"])
items = items.replace('0', np.nan)
items = items.apply(pd.to_numeric, errors='coerce')
items = items.apply(lambda x: x.fillna(.000001),axis=1)

#users dataset
users = user_pd
users = users.replace('0', np.nan)
users = users.apply(pd.to_numeric, errors='coerce')
users = users.apply(lambda x: x.fillna(.000001),axis=1)


#Some Functions including an Error function from Bugra[5]
def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)

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

y_r, Uy, Vy = svd_red(items,10)
x_R, Ux, Vx = svd_red(users, 10)


# Adapted code from Bugra[5] for ALS 
Q = movies_np

W = Q!=0
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)


lambda_ = 0.1
n_factors = 10
m, n = Q.shape
n_iterations = 20


X = 5 * Ux
Y = 5 * Uy.T

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

#Rounding and saving
weighted_Q_hat = np.round(weighted_Q_hat, decimals = 3)
weighted_errors = np.round(weighted_errors, decimals = 3)

np.savetxt("Q_weighted.csv", weighted_Q_hat, fmt='%1.3f', delimiter=",")
np.savetxt("Q_weighted_err.csv", weighted_errors, fmt='%1.3f', delimiter=",")
