# 导入所需库
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
# 读取Excel文件
file_path = "光谱特征参数+Cd和Chl加组名34.xlsx"  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 提取光谱特征和Chl和Cd列
spectral_features = data.iloc[:, :21]
Chl = data['Chl']
Cd = data['Cd']

# Function to perform hierarchical clustering for each feature with Chl or Cd and return sorted clusters and distances
def hierarchical_clustering_with_target(target_column, target_name):
    clusters = []
    distances = []
    for feature_name in spectral_features.columns:
        # Combining the spectral feature with Chl or Cd
        combined_data = pd.DataFrame({feature_name: spectral_features[feature_name], target_name: target_column})

        # Performing hierarchical clustering
        Z = linkage(combined_data, method='ward')

        # Extracting the last clustering step (largest distance)
        last_cluster = Z[-1]
        distance = last_cluster[2]

        # Adding the feature name and distance
        clusters.append(feature_name)
        distances.append(distance)

    return clusters, distances


# Performing hierarchical clustering with Chl and sorting by distances
clusters_with_Chl, distances_with_Chl = hierarchical_clustering_with_target(Chl, 'Chl')
sorted_clusters_with_Chl = [x for _, x in sorted(zip(distances_with_Chl, clusters_with_Chl), reverse=True)]
sorted_distances_with_Chl = sorted(distances_with_Chl, reverse=True)

# Performing hierarchical clustering with Cd and sorting by distances
clusters_with_Cd, distances_with_Cd = hierarchical_clustering_with_target(Cd, 'Cd')
sorted_clusters_with_Cd = [x for _, x in sorted(zip(distances_with_Cd, clusters_with_Cd), reverse=True)]
sorted_distances_with_Cd = sorted(distances_with_Cd, reverse=True)

# Creating DataFrame A with the sorted clustering results
A_clustering_sorted_final = pd.DataFrame({
    'Spectral Feature with Chl': sorted_clusters_with_Chl,
    'Distance with Chl': sorted_distances_with_Chl,
    'Spectral Feature with Cd': sorted_clusters_with_Cd,
    'Distance with Cd': sorted_distances_with_Cd
})

A_clustering_sorted_final
