from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset
data = pd.read_csv(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment3\games.csv")

# Encode categorical features if needed
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

# Split into features and true labels (for comparison only)
X = data.iloc[:, :-1]
true_labels = data.iloc[:, -1]

# KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# EM Clustering (Gaussian Mixture)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Compare clustering results with ground truth using Adjusted Rand Index
ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
ari_gmm = adjusted_rand_score(true_labels, gmm_labels)

print("Adjusted Rand Index for K-Means:", ari_kmeans)
print("Adjusted Rand Index for EM (GMM):", ari_gmm)

# Interpret the results
if ari_gmm > ari_kmeans:
    print("EM (GMM) clustering gives better results based on Adjusted Rand Index.")
elif ari_kmeans > ari_gmm:
    print("K-Means clustering gives better results based on Adjusted Rand Index.")
else:
    print("Both clustering algorithms performed similarly.")
