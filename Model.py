import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy import stats
#file = "C:\Users\lenovo\PycharmProjects\Wheat kernel classification\Seed_Data.csv"

#file_path: str = os.path.join(file)
df = pd.read_csv("Seed_Data.csv")
df = df.drop('target', axis=1)
df.info()
df.head()
X = df.iloc[:, [0,1,2,3,4,5,6]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Numbers of Cluster")
plt.ylabel("WCSS")
plt.show()
kmeans_3n = KMeans(n_clusters=3, init='k-means++')
y_3n = kmeans_3n.fit_predict(X)
print(y_3n)
kmeans_4n = KMeans(n_clusters=4, init='k-means++')
y_4n = kmeans_4n.fit_predict(X)
print(y_4n)
print(kmeans_3n.cluster_centers_)
print(kmeans_4n.cluster_centers_)
print(kmeans_3n.cluster_centers_[:, 0])

plt.scatter(X[y_3n==0,0], X[y_3n==0, 1], s=50, c='red', label='cluster1')
plt.scatter(X[y_3n==1,0], X[y_3n==1, 1], s=50, c='blue', label='cluster2')
plt.scatter(X[y_3n==2,0], X[y_3n==2, 1], s=50, c='cyan', label='cluster1')

plt.scatter(kmeans_3n.cluster_centers_[:,0], kmeans_3n.cluster_centers_[:,1], s=200, c='yellow',label="centroid")

plt.title("Clusters of Wheat Kernel")
plt.xlabel("Area")
plt.ylabel("Coefficient Assymetry")
plt.show()

plt.scatter(X[y_3n==0,0], X[y_3n==0, 2], s=50, c='red', label='cluster1')
plt.scatter(X[y_3n==1,0], X[y_3n==1, 2], s=50, c='blue', label='cluster2')
plt.scatter(X[y_3n==2,0], X[y_3n==2, 2], s=50, c='cyan', label='cluster1')

plt.scatter(kmeans_3n.cluster_centers_[:,0], kmeans_3n.cluster_centers_[:,2], s=200, c='yellow',label="centroid")

plt.title("Clusters of Wheat Kernel")
plt.xlabel("Area")
plt.ylabel("kernel_groove_length")
plt.show()

print(kmeans_4n.cluster_centers_[:,0])

plt.scatter(X[y_4n==0,0], X[y_4n==0, 1], s=50, c='red', label='cluster1')
plt.scatter(X[y_4n==1,0], X[y_4n==1, 1], s=50, c='blue', label='cluster2')
plt.scatter(X[y_4n==2,0], X[y_4n==2, 1], s=50, c='cyan', label='cluster3')
plt.scatter(X[y_4n==3,0], X[y_4n==3, 1], s=50, c='yellow', label='cluster4')

plt.scatter(kmeans_4n.cluster_centers_[:,0], kmeans_4n.cluster_centers_[:,1], s=200, c='green',label="centroid")
plt.show()

plt.scatter(X[y_4n==0,0], X[y_4n==0, 2], s=50, c='red', label='cluster1')
plt.scatter(X[y_4n==1,0], X[y_4n==1, 2], s=50, c='blue', label='cluster2')
plt.scatter(X[y_4n==2,0], X[y_4n==2, 2], s=50, c='cyan', label='cluster3')
plt.scatter(X[y_4n==3,0], X[y_4n==3, 2], s=50, c='yellow', label='cluster4')

plt.scatter(kmeans_4n.cluster_centers_[:,0], kmeans_4n.cluster_centers_[:,2], s=200, c='green',label="centroid")
plt.show()

plt.scatter(X[y_4n==0,1], X[y_4n==0, 2], s=50, c='red', label='cluster1')
plt.scatter(X[y_4n==1,1], X[y_4n==1, 2], s=50, c='blue', label='cluster2')
plt.scatter(X[y_4n==2,1], X[y_4n==2, 2], s=50, c='cyan', label='cluster3')
plt.scatter(X[y_4n==3,1], X[y_4n==3, 2], s=50, c='yellow', label='cluster4')

plt.scatter(kmeans_4n.cluster_centers_[:,1], kmeans_4n.cluster_centers_[:,2], s=200, c='green',label="centroid")
plt.show()

sil_avg= []
for i in range(2, 11):
    cluster_model = KMeans(n_clusters=i, init="k-means++")
    cluster_labels = cluster_model.fit_predict(X)
    sil_avg_score = silhouette_score(X, cluster_labels)
    sil_avg.append(sil_avg_score)
print(sil_avg)

y_gauss = GaussianMixture(n_components=3, random_state=42,).fit(X).predict(X)

print(y_gauss)

mapping = {}
for class_id in np.unique(y_gauss):
    mode, _, = stats.mode(y_gauss[y_gauss==class_id])
    print(mode)
    mapping[mode[0]] = class_id
print(mapping)

np.unique(y_gauss)

y_pred = np.array([mapping[cls] for cls in y_gauss])
print(y_pred)

plt.plot(X[y_pred==0, 0], X[y_pred==0,1], 'yo', label='cluster1')
plt.plot(X[y_pred==1, 0], X[y_pred==1,1], 'bs', label='cluster2')
plt.plot(X[y_pred==2, 0], X[y_pred==2,1], 'g^', label='cluster3')
plt.show()

plt.plot(X[y_pred==0, 0], X[y_pred==0,2], 'yo', label='cluster1')
plt.plot(X[y_pred==1, 0], X[y_pred==1,2], 'bs', label='cluster2')
plt.plot(X[y_pred==2, 0], X[y_pred==2,2], 'g^', label='cluster3')
plt.show()

plt.plot(X[y_pred==0, 1], X[y_pred==0,2], 'yo', label='cluster1')
plt.plot(X[y_pred==1, 1], X[y_pred==1,2], 'bs', label='cluster2')
plt.plot(X[y_pred==2, 1], X[y_pred==2,2], 'g^', label='cluster3')
plt.show()

np.sum(y_pred==y_gauss)
print("Accuracy:", accuracy_score(y_3n,y_pred))
print("Accuracy:", accuracy_score(y_4n,y_pred))

pickle.dump(kmeans_3n, open("model.pkl", "wb"))




