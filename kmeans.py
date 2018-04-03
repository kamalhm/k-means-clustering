import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataTrain = pd.read_csv('TrainingSet.txt', delimiter='\t').as_matrix()
print(dataTrain)

plt.scatter(dataTrain[:,0],dataTrain[:,1]) # dataTrain[:,0] mengambil kolom X, sedangkan [:,1] mengambil kolom Y
# plt.title('Scatter Plot')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
kmeans = KMeans.fit()