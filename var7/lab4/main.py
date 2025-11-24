import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import MDS, TSNE
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# dbn - school id number -> перевести в категориальное число
# enrolled = сколько учеников в целом
# present - присутствующие сегодня
# absent  - отсутствующие сегодня

# https://www.kaggle.com/datasets/sahirmaharajj/school-student-daily-attendance

data = pd.read_csv("attendance.csv", delimiter=",")
data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")

print(data.head())
print("Размер датасета:", data.shape)
print("Пропуски до очистки:\n", data.isna().sum())

data = data.dropna().reset_index(drop=True)
print("Пропуски после очистки:\n", data.isna().sum())


data["att_rate"] = data["Present"] / data["Enrolled"]  # процент присутствия
data["abs_rate"] = data["Absent"] / data["Enrolled"]   # процент отсутствия
data["rel_rate"] = data["Released"] / data["Enrolled"] # процент вышедших

le = LabelEncoder()

# преобразование категориальных данных в числовые (ID школы)
data['School DBN'] = le.fit_transform(data['School DBN'])
print(data.iloc[[4, 500, 1500]][["School DBN", "Enrolled", "Present", "Absent", "Released"]])

feature_cols = ["Enrolled", "Absent", "Present", "Released",
                "att_rate", "abs_rate", "rel_rate"]

data_train = data[feature_cols]

print("Первые строки матрицы признаков:\n", data_train.head())

# масштабирование данных
scaler = StandardScaler()
X = data_train.values
X_scaled = scaler.fit_transform(X)

np.random.seed(42)
sample_size = min(20000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)
X_sample = X_scaled[sample_idx]

inertia = []
sil_scores = []

for i in range(2, 15):  # с k=1 silhouette не считается, поэтому с 2
    km = KMeans(n_clusters=i, random_state=0, max_iter=300, n_init=10)
    labels_k = km.fit_predict(X_sample)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sample, labels_k))  # силуэт

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(2, 15), inertia, marker='o')
plt.title('метод локтя')
plt.xlabel('кол-во кластеров k')
plt.ylabel('инерция (SSE внутри кластеров)')

plt.subplot(1, 2, 2)
plt.plot(range(2, 15), sil_scores, marker='o')
plt.title('silhouette score')
plt.xlabel('кол-во кластеров k')
plt.ylabel('silhouette')

plt.tight_layout()
plt.show()

k = 5 
kmeans = KMeans(n_clusters=k, random_state=0, max_iter=300, n_init=10)
kmeans.fit(X_scaled)

# предсказанные метки кластера для всех объектов
labels = kmeans.labels_
data["cluster"] = labels

np.random.seed(42)
idx = np.random.choice(len(X_scaled), size=sample_size, replace=False)

X_sub = X_scaled[idx]
labels_sub = labels[idx]

# силуэтный коэффициент
print(f"силуэтный коэффициент: {silhouette_score(X_sub, labels_sub):.4f}")

# оценка davies-bouldin
print(f"оценка davies-bouldin: {davies_bouldin_score(X_sub, labels_sub):.4f}")

# индекс Калински-Харабаса
print(f"индекс калински-харабаса: {calinski_harabasz_score(X_sub, labels_sub):.2f}")

within_ss = kmeans.inertia_
centroids = kmeans.cluster_centers_
between_ss = 0.0
for i in range(k):
    for j in range(i + 1, k):
        between_ss += np.sum((centroids[i] - centroids[j]) ** 2)

print(f"SSE внутри кластеров (inertia): {within_ss:.2f}")
print(f"сумма квадратов расстояний между центроидами: {between_ss:.2f}")

models = [
    ("PCA", PCA(n_components=2)),
    ("FastICA", FastICA(n_components=2, random_state=0)),
    ("TruncatedSVD", TruncatedSVD(n_components=2)),
    ("MDS", MDS(n_components=2, random_state=0)),
    ("TSNE", TSNE(n_components=2, random_state=0)),
    ("UMAP", UMAP(n_components=2, random_state=0))
]

# делаем отдельную выборку, потому что датасет пизда огромный)
np.random.seed(42)
vis_size = min(5000, len(X_scaled))
vis_idx = np.random.choice(len(X_scaled), size=vis_size, replace=False)
X_vis = X_scaled[vis_idx]
labels_vis = labels[vis_idx]
schools_vis = data['School DBN'].values[vis_idx]

for name, model in models:
    data_transformed = model.fit_transform(X_vis)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(data_transformed[:, 0], data_transformed[:, 1],
                   c=labels_vis, cmap='rainbow', s=5)
    axs[0].set_title(f'{name} - кластеры k-means')

    axs[1].scatter(data_transformed[:, 0], data_transformed[:, 1],
                   c=schools_vis, cmap='rainbow', s=5)
    axs[1].set_title(f'{name} - распределение по школам')

    plt.tight_layout()
    plt.show()