import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score, 
    adjusted_rand_score
    )
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import MDS, TSNE
from umap.umap_ import UMAP
import matplotlib.pyplot as plt

# цель - кластеризовать
df = pd.read_csv("Mall_Customers.csv", delimiter=",")
print(df.head(), df.describe(), df.shape)
print(df.info())
print(df.isna().sum())

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
# 1 male 0 female
print(df.iloc[[4, 50, 130]].head())

train_df = df.drop("Gender", axis=1)
print(train_df.head())

#метод локтя
inertia = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(train_df)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 15), inertia, marker='o')
plt.title('метод локтя')
plt.xlabel('колво кластеров')
plt.ylabel('инерция')
plt.show()

clusters=5

kmeans = KMeans(n_clusters=clusters, random_state=0, max_iter=300)
kmeans.fit(train_df)

labels = kmeans.labels_
print(f"силуэтный коэф avg: {silhouette_score(train_df, labels)}") # 0.35833526645086095
print(f"davies bouldin avg: {davies_bouldin_score(train_df, labels)}") # 0.9421168182600759
print(f"kalinski harabasz avg: {calinski_harabasz_score(train_df, labels)}") # 240.14988869196696
print(f'Adjusted Rand Index: {adjusted_rand_score(df['Gender'], kmeans.labels_)}') #-0.002440444561370305

# модели для уменьшения размерности
models = [
    ("PCA", PCA(n_components=2)),
    ("FastICA", FastICA(n_components=2, random_state=0)),
    ("TruncatedSVD", TruncatedSVD(n_components=2)),
    ("MDS", MDS(n_components=2, random_state=0)),
    ("TSNE", TSNE(n_components=2, random_state=0)),
    ("UMAP", UMAP(n_components=2, random_state=0))
]

for name, model in models:
    data_transformed = model.fit_transform(train_df)

    figure, axis = plt.subplots(1, 2, figsize=(12, 6))

    axis[0].scatter(data_transformed[:, 0], data_transformed[:, 1], c=kmeans.labels_, cmap='rainbow')
    axis[0].set_title(f'{name} - Предсказанные категории')

    # График с оригинальными данными
    # Важно заменить 'data['Species']' на соответствующую переменную, содержащую истинные метки
    axis[1].scatter(data_transformed[:, 0], data_transformed[:, 1], c=df['Gender'], cmap='rainbow')
    axis[1].set_title(f'{name} - Оригинальный датасет')

    plt.show()
