import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

df = pd.read_csv('iris.csv') #load data
data = np.delete(np.array(df), 4, axis=1)

max_iter = 500 #max iteration of the centroid calculation
z = 1 #for plotting at L.29
for k in range(2,6):
	index = np.array([0]*150)
	#select initial centroids as random data vectors
	initial_c = random.sample(range(151), k)
	#initial_c = np.random.randint(0,150,k)
	for i in range(len(initial_c)):
		index[initial_c[i]] = i
	for i in range(max_iter): #centroids update
		C = np.array([np.mean(data[index==j],axis=0) for j in range(k)]) #calculate centroids
		new_index = [np.argmin([np.linalg.norm(x-c)**2 for c in C]) for x in data]
		if (all(index == np.array(new_index))):
			break
		index = np.array(new_index)
	#4-dimensional -> 2-dimensional data by t-SNE
	tsne_result = TSNE(n_components=2).fit_transform(data)
	tsne_center = np.array([np.mean(tsne_result[index==j],axis=0) for j in range(k)]) #calculate of centroids from the 2D data

	colorlist = ["r", "g", "b", "c", "m"] #plotting
	plt.subplot(2,2,z)
	for i in range(k):
		a = np.where(index==i)[0]
		print (len(a))
		#plt.scatter(data[a][:,0], data[a][:,3], c=colorlist[i])
		plt.scatter(tsne_result[a][:,0], tsne_result[a][:,1], c=colorlist[i])
		plt.scatter(tsne_center[i,0], tsne_center[i,1], marker='x', s=80, c='black', linewidths="3", edgecolors='black')
	z += 1
plt.show()
