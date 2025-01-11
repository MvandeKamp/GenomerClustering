import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 1. Daten laden
def load_data(data_path, labels_path):
    # Daten und Labels laden
    data = pd.read_csv(data_path, index_col=0)  # Annahme: Erste Spalte ist der Index
    labels = pd.read_csv(labels_path, index_col=0)  # Annahme: Labels in separater Datei
    return data.values, labels.values.ravel()

# Lokale Pfade zu den Daten
data_path = r"../TCGA-PANCAN-HiSeq-801x20531\data.csv"
labels_path = r"../TCGA-PANCAN-HiSeq-801x20531\labels.csv"

# Daten und Labels laden
data, true_labels = load_data(data_path, labels_path)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 2. PCA-Analyse
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

pca_data, pca_model = apply_pca(data_scaled)

# 3. Autoencoder-Analyse
from tensorflow.keras.layers import LeakyReLU

def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder

encoding_dim = 2  # Wir reduzieren auf 2 Dimensionen
autoencoder, encoder = build_autoencoder(data_scaled.shape[1], encoding_dim)

# Autoencoder trainieren
autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=32, shuffle=True, verbose=0)

# Daten mit dem Encoder transformieren
autoencoded_data = encoder.predict(data_scaled)

# 4. Clustering mit KMeans
def apply_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    return cluster_labels, silhouette_avg

# Clustering auf PCA-Daten
pca_clusters, pca_silhouette = apply_kmeans(pca_data)

# Clustering auf Autoencoder-Daten
autoencoder_clusters, autoencoder_silhouette = apply_kmeans(autoencoded_data)

# 5. Visualisierung
def plot_clusters(data, cluster_labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

# PCA-Visualisierung
plot_clusters(pca_data, pca_clusters, f'PCA Clustering (Silhouette: {pca_silhouette:.2f})')

# Autoencoder-Visualisierung
plot_clusters(autoencoded_data, autoencoder_clusters, f'Autoencoder Clustering (Silhouette: {autoencoder_silhouette:.2f})')

# 6. Vergleich der Silhouette-Werte
print(f"Silhouette-Wert für PCA: {pca_silhouette:.2f}")
print(f"Silhouette-Wert für Autoencoder: {autoencoder_silhouette:.2f}")