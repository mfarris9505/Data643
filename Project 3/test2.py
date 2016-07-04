# -*- coding: utf-8 -*-
"""
Created on Sun Jul 03 19:49:07 2016

@author: Matts42
"""

def svd_red(U1,s1,V1,n):
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

def similarity(mat):
    item_similarity = 1 - pairwise_distances(mat, metric='cosine')
    return item_similarity

R, Uk, Vk = svd_red(U1,s1,V1,100)

sim_Uk = similarity(Uk)
sim_Vk = similarity(Vk)


def final_pred(R,similarity):
    pred = R.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])  
    final_pred = pred
    final_pred += user_mean
    return final_pred


final_item = final_pred(R,sim_Vk.T)
final_user = final_pred(R,sim_Vk.T)
