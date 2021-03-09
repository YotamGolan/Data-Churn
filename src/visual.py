import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def information_gained_bar():
    intake_data = pd.read_csv('Customer_Churn_processed.csv')
    data_leave = intake_data['LEAVE']
    intake_data.drop(columns = ['LEAVE'], inplace = True)

    info_gain = list(zip(list(intake_data.columns), mutual_info_classif(intake_data, data_leave, discrete_features=True)))
    info_gain = sorted(info_gain, key=lambda x: x[1], reverse=True)
    (labels, info_new) = zip(*info_gain)

    #'HOUSE','INCOME', 'OVERAGE', 'HANDSET', 'OVER_15MIN', 'LEFTOVER', 'AVG_CALL', 'REPORT_SATIS', 'COLLEGE", 'CHANGE_PLAN', 'REP_USAGE'
    plt.bar(['A', 'B', 'C', 'D', 'E', 'F', "G", "H", "I", "J", "K"], list(info_new))
    for i, v in enumerate(list(info_new)):
        plt.text(i - 0.45, v + 0.01, str('%.3f'%(v)))
    plt.xlabel('Attributes', fontweight='bold')
    plt.ylabel('Information Gain', fontweight='bold')
    plt.title('Attributes ranked by Information Gain')
    plt.show()

def correlation_matrix():
    intake_data = pd.read_csv('Customer_Churn_processed.csv')
    corr_data = intake_data.corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', xticklabels=['College', 'Income', 'Overage', 'Leftover', 'House', 'Handset', 'Over15', 'AvgCallDur', 'RepSatis', 'RepUsage', 'ConsidChange', 'Leave'],
                yticklabels=['College', 'Income', 'Overage', 'Leftover', 'House', 'Handset', 'Over15', 'AvgCallDur', 'RepSatis', 'RepUsage', 'ConsidChange', 'Leave'])
    plt.show()
    #print(corr_data)

def cluster_selection():
    intake_data = intake_data = pd.read_csv('Customer_Churn_processed.csv')
    intake_data = intake_data.to_numpy()
    scaled_data = StandardScaler().fit_transform(intake_data)

    k_means = KMeans(n_clusters = 5, init = 'k-means++')
    k_means.fit(scaled_data)
    data_centers = k_means.cluster_centers_
    labels = k_means.labels_

    test_clusters = {0: [], 1: [], 2: [], 3: [], 4: []}
    for i, test_cluster in enumerate(labels):
        test_clusters[test_cluster].append(intake_data[i].tolist())
        #print(test_cluster)
    df_clusters = []
    #print(test_cluster)
    for i in range(5):
        df_clusters.append(pd.DataFrame(test_clusters[i]))
    #print(df_clusters)

    for i, df in enumerate(df_clusters):
        print('Cluster {}'.format(i+1))
        print('=================================================')
        print(df.describe())
        #print(df.head())
        print('================================================= \n')


#information_gained_bar()    
#correlation_matrix()
cluster_selection() #I found 5 to work well
