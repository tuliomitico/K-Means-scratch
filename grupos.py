import numpy as np
from sklearn.cluster import MeanShift,DBSCAN,KMeans,AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs,make_circles
import pandas as pd
from Clusters.kmeans import KMedias

iris = pd.read_csv("iris_data.csv",names=["comprimento sepala (cm)","largura sepala (cm)",
         "comprimento petala (cm)","largura petala (cm)","classes"] )

iris_sem_t = iris.drop(["classes"],axis=1) 

def plot_agglo(X):
    agg = AgglomerativeClustering(n_clusters=3)
    labels = agg.fit_predict(X)
    
    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=labels,cmap="viridis")
    plt.colorbar()
    plt.show()

def plot_db(X):
    escala = StandardScaler()
    escala.fit(X)
    x_escalado=escala.transform(X)
    
    db = DBSCAN(eps=0.6,min_samples=1.985)
    grupos = db.fit_predict(x_escalado)
    
    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=grupos,cmap="viridis")
    plt.colorbar()
    plt.show()

def plot_kmedias(X):
    kmed = KMedias(X,n_grupos=3,itera_max=500,nr_init_centroides=1)
    kmed.encaixe()
    prev = kmed.previsto(X)
    centros = kmed.centros_clusters()
    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=prev,cmap="viridis")
    plt.scatter(centros[:,0],centros[:,1],marker="*",
                color="midnightblue",s=20,linewidths=5,zorder = 10)
    plt.show()

def plot_kmeans(X):
    km = KMeans(n_clusters=3,max_iter=500,random_state=0)
    km.fit(X)
    pred = km.predict(X)
    centros = km.cluster_centers_
    plt.scatter(X.iloc[:,0],X.iloc[:,1],c = pred,cmap = "viridis")
    plt.scatter(centros[:,0],centros[:,1],
                marker="^",color = "red",s = 20, linewidths =5, zorder = 10)
    plt.show()

def plot_mean(X):
    ms = MeanShift()
    ms.fit(X)
    labels = ms.labels_
    centro_grupos = ms.cluster_centers_
    plt.scatter(X.iloc[:,0],X.iloc[:,1],c=labels,cmap="viridis")
    plt.scatter(centro_grupos[:,0],centro_grupos[:,1],
                marker = "o",color = 'k', s = 20, linewidths = 5, zorder = 10)
    plt.show()
    
#plt.subplot(121)
#plot_agglo(iris_sem_t)
#plt.subplot(122)
#plot_db(iris_sem_t)
#plt.subplot(123)
#plot_mean(iris_sem_t)
#plt.subplot(131)
plot_kmeans(iris_sem_t)
plot_kmedias(iris_sem_t)
