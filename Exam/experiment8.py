from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_iris

dataset=load_iris()
x=dataset.data
y=dataset.target

k_mean=KMeans(n_clusters=2,random_state=42)
k_mean_pred=k_mean.fit_predict(X=x)

gmm=GaussianMixture(n_components=2,random_state=42)
gmm_pred=gmm.fit_predict(x)

k_mean_score=adjusted_rand_score(k_mean_pred,y)
gmm_score=adjusted_rand_score(gmm_pred,y)

if(k_mean_score>gmm_score):
    print("K Mean clustering perform better than EM on adjusted rand score")
elif(k_mean_score<gmm_score):
    print("GMM clustering perform better than KMean on adjusted rand score")
else:
    print("Both clustering perform similarly on adjusted rand score")
