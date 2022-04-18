import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = np.array([[0,0,0,0,0],
    [9,0,0,0,0],
    [3,7,0,0,0],
    [6,5,9,0,0],
    [11,10,2,8,0],])
    

linked = linkage(X, 'single')

labelList = range(1, 6)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='ascending',
            show_leaf_counts=True)
plt.show()
