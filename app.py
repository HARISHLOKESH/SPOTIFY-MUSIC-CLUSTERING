import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("🎧 Spotify Music Clustering")
st.write("Cluster songs based on audio characteristics like Danceability, Energy, Tempo, Loudness and Valence.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/zaheenhamidani/ultimate-spotify-tracks-db/master/SpotifyFeatures.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Missing Values")
st.write(df.isnull().sum())


st.subheader("Selected Features for Clustering")

features = df[['danceability','energy','tempo','loudness','valence']]
st.write(features.head())


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


st.subheader("Elbow Method (Optimal Clusters)")

inertia = []

for k in range(1,10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(range(1,10), inertia, marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")

st.pyplot(fig1)


st.subheader("KMeans Clustering")

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

st.write("Cluster Distribution")
st.write(df['cluster'].value_counts())


st.subheader("Cluster Visualization using PCA")

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(pca_result, columns=['PC1','PC2'])
pca_df['cluster'] = df['cluster']

fig2, ax2 = plt.subplots(figsize=(8,6))

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="Set1",
    data=pca_df,
    ax=ax2
)

ax2.set_title("Spotify Song Clusters (PCA)")

st.pyplot(fig2)


st.subheader("Cluster Feature Insights")

cluster_summary = df.groupby('cluster')[features.columns].mean()

st.write(cluster_summary)
