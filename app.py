import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title("Spotify Music Clustering")

st.write("Cluster songs based on audio characteristics")

# Load dataset
df = pd.read_csv("data/SpotifyFeatures.csv")

st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Features for clustering
features = df[['danceability','energy','tempo','loudness','valence']]

# Normalize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

st.subheader("Cluster Distribution")
st.write(df['cluster'].value_counts())

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(pca_result, columns=['PC1','PC2'])
pca_df['cluster'] = df['cluster']

# Plot
st.subheader("Cluster Visualization (PCA)")

fig, ax = plt.subplots()

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="Set1",
    data=pca_df,
    ax=ax
)

st.pyplot(fig)
