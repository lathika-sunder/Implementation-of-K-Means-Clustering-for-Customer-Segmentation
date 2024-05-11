## Implementation-of-K-Means-Clustering-for-Customer-Segmentation
### DATE:02.04.2024
### AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
1. Import the necessary  packages 
2. Insert the dataset to perform the k - means clustering
3. perform k - means clustering on the dataset 
4. Then print the centroids and labels 
5. Plot graph and display the clusters 

### Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Lathika Sunder
RegisterNumber:212221230054
```
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers_EX8.csv")
data
X=data[["Annual Income (k$)","Spending Score (1-100)"]]
X
plt.figure(figsize=(4,4))
plt.scatter(data["Annual Income (k$)"],data["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
k =5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points = X[labels==i]
  plt.scatter(cluster_points["Annual Income (k$)"],cluster_points["Spending Score (1-100)"],
              color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:,0],centroids[:,1],marker="*",s=200,color='k',label='Centroids')
plt.title("K- means Clustering")
plt.xlabel("Annual Incme (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
```
### Output:
- INSERTED DATASET:
![image](https://github.com/gpavana/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787343/30c21ad5-6165-443a-9c72-0619abd15d2f)
- CENTROID AND LABEL VALUES:
![image](https://github.com/gpavana/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787343/e569502b-a75e-40bb-9183-af84a64e61b1)
- CLUSTERING:
![image](https://github.com/gpavana/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/118787343/6a0de927-a165-43f3-a5af-999dd3b9cd2b)
### Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
