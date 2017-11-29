#!/usr/bin/env python3

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import load

(X_train, y_train, X_test) = load.train_test()

scaler = MinMaxScaler(feature_range=(-1,1))

X_train = scaler.fit_transform(X_train)


### TSNE ###
tsne = TSNE()
X_tsne = tsne.fit_transform(X_train)
plt.figure(1,figsize=(10,10))

for col, i, t_name in zip(['blue','red'], [0,1], [0,1]):
    print(i)
    plt.scatter(X_tsne[y_train == i, 0], X_tsne[y_train == i, 1], color=col, s=1, alpha=0.8, label=t_name, marker='.')

plt.show()

### PCA ###

n_components = 30

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_train)
print('var explained', pca.explained_variance_ratio_.sum())

for i in range(n_components):
    print(pca.explained_variance_ratio_[i])
    
plt.figure(2)

for color, i, t_name in zip(['blue','red'], [0,1], [0,1]):
    plt.scatter(X_pca[y_train == i, 0], X_pca[y_train == i, 1], color=color, s=1, alpha=0.8, label=t_name, marker='.')
plt.show()