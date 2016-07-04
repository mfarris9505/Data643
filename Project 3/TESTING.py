# -*- coding: utf-8 -*-
"""
Created on Sat Jul 02 23:02:49 2016

@author: Matts42
"""

import pandas as pd 
import numpy as np
import math
import csv
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from scipy.spatial.distance import cosine


#Data file containing ratings data
f = open("data/Movies.txt")
text = csv.reader(f, delimiter = "\t")
ratings = list(text)
myMovieRatings = pd.DataFrame(ratings, columns = ["Movie_id", "Title"])
rate = [4,5,5,4,2,2,2,1,4,4,5,3,2,2,3,4,2,5,2,1]
myratings = pd.DataFrame(rate, columns =["Rating"])
dfs = [myMovieRatings,myratings]
myMovieRatings = pd.concat(dfs, axis = 1)
#Data file containing ratings data
f = open("data/u.data")
text = csv.reader(f, delimiter = "\t")
ratings = list(text)

# My Ratings of the movie Data (I left out the ones I wanted to "guess")
my_ratings = [4,5,5,4,0,2,2,1,4,4,5,3,2,2,3,0,2,5,2,1]

ratings_pd = pd.DataFrame(ratings, columns = ["user_id","movie_id", "rating","timestamp"])
ratings_pd = ratings_pd.drop(["timestamp"], 1)

movies = ratings_pd.pivot(index = "movie_id", columns = "user_id", values = "rating")

#Movie IDs
i = ['55','56','64','71','94','102','118','133','141','168','173','196','1278','928','780',
     '651','514','483','432','404']
movies = movies.loc[i,:]

#Random User IDs
l = ['13','128','201','207','222','23','234','246','267','268','269','276','493','642',
     '655','846','95','262','194','130','1']
movies = movies.loc[:,l]
movies = movies.fillna(value=0)

movies_np = np.array(movies)
movies_np = movies_np.astype(float)
movies_np = np.insert(movies_np, 0, my_ratings, axis=1)


def norm_mean_cent(movies_np):
    mean_movie= []
    
    count_movie = []

    for row in movies_np:
        row_sum = np.sum(row)
        count = np.count_nonzero(row)
        count_movie.append(count)
        mean_movie.append(row_sum/count)
    
    count_user = []
    mean_user = []

    for row in movies_np.T:
        row_sum = np.sum(row)
        count = np.count_nonzero(row)
        count_user.append(count)
        mean_user.append(row_sum/count)

    movies_np[movies_np==0] = np.nan

    mean_cent = []
    i = 0
    for row in  movies_np:
        mean_cent.append(row - mean_movie[i]) 
        i += 1
    
    mean_cent = np.array(mean_cent)
    mean_cent = np.nan_to_num(mean_cent)
    
    return mean_cent

movies_np  = np.nan_to_num(movies_np)
movies_norm = preprocessing.normalize(movies_np, norm='l2')
mean_cent = norm_mean_cent(movies_np)

U1, s1, V1 = np.linalg.svd(mean_cent, full_matrices=True)

U2, s2, V2 = np.linalg.svd(movies_norm, full_matrices=False)