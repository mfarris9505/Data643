# -*- coding: utf-8 -*-
"""
Data Extraction and Cleaning

Source:http://grouplens.org/datasets/movielens/

"""
import pandas as pd 
import numpy as np
import math
import csv


# Data file containing ratings data
f = open("data/u.data")
text = csv.reader(f, delimiter = "\t")
ratings = list(text)

# My Ratings of the movie Data (I pick)
my_ratings = [4,5,5,4,0,2,2,1,4,4,5,3,2,2,3,0,2,5,2,1]

ratings_pd = pd.DataFrame(ratings, columns = ["user_id","movie_id", "rating","timestamp"])
ratings_pd = ratings_pd.drop(["timestamp"], 1)

movies = ratings_pd.pivot(index = "movie_id", columns = "user_id", values = "rating")

i = ['55','56','64','71','94','102','118','133','141','168','173','196','1278','928','780','651','514','483','432','404']
movies = movies.loc[i,:]

l = ['13','128','201','207','222','23','234','246','267','268','269','276','493','642','655','846','95','262','194','130','1']
movies = movies.loc[:,l]
movies = movies.fillna(value=0)

movies_np = np.array(movies)
movies_np = movies_np.astype(float)
movies_np = np.insert(movies_np, 0, my_ratings, axis=1)

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
U = []
 
for row in mean_cent:
    y = 0
    for val in row:
        x = math.pow(val,2)
        y = y + x
    U.append(math.sqrt(y))

Mov_Mov =[]
Mov = []

cent_t = mean_cent.T

num = np.dot(mean_cent,mean_cent.T)

for i in range(20):
    for j in range(20):
        num[i,j] = num[i,j]/(U[i]*U[j])
        
        
        