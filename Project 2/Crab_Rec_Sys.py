# -*- coding: utf-8 -*-
"""
Data Extraction and Cleaning

Source:http://grouplens.org/datasets/movielens/

"""
import pandas as pd 
import numpy as np
import csv



# Data file containing ratings data
f = open("data/u.data")
text = csv.reader(f, delimiter = "\t")
ratings = list(text)

# My Ratings of the movie Data (I pick)
my_ratings = [4,5,5,4,2,2,2,1,4,4,5,3,2,2,3,3,2,5,2,1]

ratings_pd = pd.DataFrame(ratings, columns = ["user_id","movie_id", "rating","timestamp"])
ratings_pd = ratings_pd.drop(["timestamp"], 1)

movies = ratings_pd.pivot(index = "movie_id", columns = "user_id", values = "rating")


i = ['55','56','64','71','94','102','118','133','141','168','173','196','1278','928','780','651','514','483','432','404']
movies = movies.loc[i,:]

l = ['13','128','201','207','222','23','234','246','267','268','269','276','493','642','655','846','95','262','194','130','1']
movies = movies.loc[:,l]

movies = movies.astype(float).fillna(0.0)

movie_dict = movies.to_dict()

from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import euclidean_distances
from scikits.crab.similarities import ItemSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender

model = MatrixPreferenceDataModel(movie_dict)
similarity = ItemSimilarity(model, euclidean_distances)
recommender = UserBasedRecommender(model, similarity, with_preference=True)


