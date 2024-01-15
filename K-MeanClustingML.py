import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D


url = "https://raw.githubusercontent.com/MicrosoftDocs/ml-basics/master/challenges/data/clusters.csv"
df = pd.read_csv(url)


print(df.head())


X = df[['A', 'B', 'C']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  
plt.show()


optimal_num_clusters = 3  

kmeans_model = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans_labels = kmeans_model.fit_predict(X_scaled)


df['Cluster'] = kmeans_labels



fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X['A'], X['B'], X['C'], c=kmeans_labels, cmap='viridis')
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('C')


plt.colorbar(scatter)

plt.title('KMeans Clustering')
plt.show()
