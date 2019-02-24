
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import datasets as syn_ds 


# In[3]:


from sklearn.cluster import KMeans


# In[12]:


plt.figure(figsize=(1, 2))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)


# In[13]:


plt.subplot(325)
plt.title("Three blobs", fontsize='small')
X1, Y1 = syn_ds.make_blobs(n_samples=1500,centers=3,random_state=123)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')


# In[34]:


kmeans_model = KMeans(n_clusters=3, random_state=170).fit(X1)
y_pred = kmeans_model.predict(X1)


# In[35]:


plt.subplot(326)
plt.title("Three blobs", fontsize='small')
X1, Y1 = syn_ds.make_blobs(n_samples=1500,centers=3,random_state=123)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=y_pred,
            s=25, edgecolor='k')


# In[24]:


print(X1.shape)
print(Y1.shape)


# In[25]:


Y1


# In[36]:


y_pred


# In[39]:


kmeans_model.cluster_centers_


# In[42]:


Ks = range(1, 10)
km = [KMeans(n_clusters=i,random_state=170) for i in Ks]
score = [km[i].fit(X1).score(X1) for i in range(len(km))]
plt.plot(Ks, score)


# In[92]:


max_iter = 500
centroids = X1[random.sample(range(0,X1.shape[0]),3)]
epsilon = 0.0001
for i in range(max_iter):
    clusters = {}
    for i in range(3):
        clusters[i] = []
        
    for point in X1:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster_clsfn = distances.index(min(distances))
        clusters[cluster_clsfn].append(point)
        
    previous = centroids
    for cluster in clusters:
        centroids[cluster] = np.average(clusters[cluster],axis=0)
        
    isOptimal = True
    for i in range(3):
        if np.sum(centroids[i] - previous[i]) > epsilon:
            isOptimal = False
    if isOptimal:
        break


# In[75]:


print(X1[0])
print(X1[1])
print(X1[0]-X1[1])
print(np.sum(X1[0]-X1[1]))


# In[65]:


np.linalg.norm(X1[0]-centroids[1])


# In[67]:


np.average(X1,axis=0)


# In[68]:


a={}
a[0] = X1[0:20]
a[1] = X1[21:40]
a[2] = X1[41:60]


# In[78]:


centroids


# In[94]:


for centroid in centroids:
    plt.scatter(centroids[:,0], centroids[:,1], s = 130, marker = "x")


# In[103]:


col=["r","g","b"]
for cluster in clusters:
    for point in clusters[cluster]:
        plt.scatter(point[0], point[1], color = col[cluster],s = 30,marker="o")


# In[88]:


random.sample(range(0,X1.shape[0]),3)


# In[91]:


X1[random.sample(range(0,X1.shape[0]),3)]


# In[101]:


import scipy.cluster.hierarchy as shc


# In[102]:


plt.figure(figsize=(10, 7))  
plt.title("Dendogram")  
dend = shc.dendrogram(shc.linkage(X1, method='ward'))  


# In[104]:


from sklearn.cluster import AgglomerativeClustering

h_cluster_model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit(X1)  
  


# In[106]:


y_pred_h = h_cluster_model.labels_


# In[108]:


y_pred_h.shape


# In[109]:


plt.subplot(326)
plt.title("Three blobs", fontsize='small')
X1, Y1 = syn_ds.make_blobs(n_samples=1500,centers=3,random_state=123)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=y_pred_h,
            s=25, edgecolor='k')

