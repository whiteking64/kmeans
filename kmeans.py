import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

df = pd.read_csv('iris.csv') #データの読み込み
data = np.delete(np.array(df), 4, axis=1)

max_iter = 500 #重心の計算（更新）の最高回数
z = 1 #L.27のプロット用
for k in range(2,6):
	index = np.array([0]*150)
	initial_c = np.random.randint(0,150,k) #初期の重心をランダムなデータ点とする
	for i in range(len(initial_c)):
		index[initial_c[i]] = i
	for i in range(max_iter): #更新を行う
		C = np.array([np.mean(data[index==j],axis=0) for j in range(k)]) #重心の計算
		new_index = [np.argmin([np.linalg.norm(x-c)**2 for c in C]) for x in data]
		if (all(index == np.array(new_index))):
			break
		index = np.array(new_index)
	#t-SNEで2次元データに変換
	tsne_result = TSNE(n_components=2).fit_transform(data)
	tsne_center = np.array([np.mean(tsne_result[index==j],axis=0) for j in range(k)]) #変換されたクラスタごとに重心を計算

	colorlist = ["r", "g", "b", "c", "m"] #以下，グラフの描画
	plt.subplot(2,2,z)
	for i in range(k):
		a = np.where(index==i)[0]
		print (len(a))
		#plt.scatter(data[a][:,0], data[a][:,3], c=colorlist[i])
		plt.scatter(tsne_result[a][:,0], tsne_result[a][:,1], c=colorlist[i])
		plt.scatter(tsne_center[i,0], tsne_center[i,1], marker='x', s=80, c='black', linewidths="3", edgecolors='black')
	z += 1
plt.show()

