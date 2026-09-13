import h5py
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_wsi_embeddings(h5_path):
    with h5py.File(h5_path, "r") as f:
        features = f["features"][:]
    return features


def kmeans_cluster(embeddings, n_clusters=15):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_


def save_slide_prototypes(prototypes, save_dir, slide_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{slide_name}_prototypes_kmeans.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(prototypes, f)
    print(f"Saved {save_path}")


def process_single_wsi(wsi_path, save_dir):
    if not os.path.exists(wsi_path):
        print(f"WSI path not found: {wsi_path}")
        return

    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    save_path = os.path.join(save_dir, f"{slide_name}_prototypes_kmeans.pkl")
    if os.path.exists(save_path):
        print(f"already exists: {slide_name}")
        return

    print(f"Processing {slide_name} ...")

    features = load_wsi_embeddings(wsi_path)
    prototypes = kmeans_cluster(features, n_clusters=7)
    save_slide_prototypes(prototypes, save_dir, slide_name)


def process_wsi_from_folder(h5_dir, save_dir="slide_prototypes", num_threads=8, suffix=".h5"):
    wsi_paths = [
        os.path.join(h5_dir, f)
        for f in os.listdir(h5_dir)
        if f.endswith(suffix)
    ]

    print(f"Found {len(wsi_paths)} h5 files in {h5_dir}")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_single_wsi, wsi_path, save_dir)
            for wsi_path in wsi_paths
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    process_wsi_from_folder(
        h5_dir="./TCGA-NSCLC/uni_512/h5_files",
        save_dir="./TCGA-NSCLC/slide_prototypes",
        num_threads=1
    )