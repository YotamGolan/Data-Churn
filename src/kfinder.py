import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_best_k():
    max_k = 21

    intake_data = pd.read_csv('Customer_Churn_processed.csv') 
    intake_data = intake_data.to_numpy()
    scaled_data = StandardScaler().fit_transform(intake_data)

    sse = []
    sil = []

    for k in range(2, max_k):
        print('Training begun for k-means for k = {}'.format(k))
        k_mean = KMeans(n_clusters=k, init='k-means++')
        k_mean.fit(scaled_data)
        sse.append(k_mean.inertia_)
        sil.append(silhouette_score(scaled_data, k_mean.labels_))


    #plot Sum Square Error
    frame = pd.DataFrame({'Cluster': range(2, max_k), 'SSE': sse})
    plt.figure(figsize=(10, 5))
    plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    plt.xticks(range(2,max_k))
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum Squared Error')

    plt.show()

    #plot silhouette
    frame = pd.DataFrame({'Cluster': range(2, max_k), 'Silhouette': sil})
    plt.figure(figsize=(10, 5))
    plt.plot(frame['Cluster'], frame['Silhouette'], marker='o')
    plt.xticks(range(2, max_k))
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette')

    plt.show()

find_best_k()