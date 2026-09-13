import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F


def load_all_slide_prototypes(prototypes_dir, test_slide_ids=None):
    if test_slide_ids is None:
        test_slide_ids = set()

    all_prototypes = []
    kept_files = []

    file_list = sorted(
        [f for f in os.listdir(prototypes_dir) if f.endswith("_prototypes_kmeans.pkl")]
    )

    for fname in file_list:
        slide_id = fname.split("_prototypes")[0]

        if slide_id in test_slide_ids:
            print(f"[SKIP TEST] {fname}")
            continue

        with open(os.path.join(prototypes_dir, fname), "rb") as f:
            slide_prototypes = pickle.load(f)

        all_prototypes.append(slide_prototypes)
        kept_files.append(fname)
        print(f"Loaded {fname}, shape={slide_prototypes.shape}")

    if len(all_prototypes) == 0:
        raise RuntimeError("No prototype files loaded.")

    all_prototypes = np.vstack(all_prototypes)
    print(f"✔ Total loaded prototypes: {all_prototypes.shape}")
    return all_prototypes, kept_files



def cluster_dataset_prototypes(all_prototypes, n_clusters=15):
    print(f"Clustering into {n_clusters} dataset prototypes ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    kmeans.fit(all_prototypes)
    return kmeans.cluster_centers_



def encode_text_features(json_path, bert_path, device="cpu"):
    with open(json_path, "r") as f:
        text_data = json.load(f)

    texts = list(text_data.values())[0]

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BertModel.from_pretrained(bert_path).to(device)
    bert.eval()

    text_features = []
    with torch.no_grad():
        for t in texts:
            inputs = tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                max_length=128
            ).to(device)
            cls_emb = bert(**inputs).last_hidden_state[:, 0, :]
            text_features.append(cls_emb.cpu())

    text_features = torch.cat(text_features, dim=0)
    print(f"✔ Loaded text features: {text_features.shape}")
    return text_features


def fuse_prototypes_with_text_concat(dataset_prototypes, text_features):
    proto = torch.from_numpy(dataset_prototypes).float()
    text = text_features.float()

    torch.manual_seed(42)
    proj = F.normalize(torch.randn(768, 1024), dim=0)
    text_1024 = text @ proj

    proto_norm = F.normalize(proto, dim=1)
    text_norm = F.normalize(text_1024, dim=1)


    sim = proto_norm @ text_norm.t()

    weights = F.softmax(sim, dim=1)
    weighted_text = weights @ text_1024

    fused = torch.cat([proto, weighted_text], dim=1)

    print(f"✔ Fused shape: {fused.shape}")
    print(f"✔ Sim matrix shape: {sim.shape}")

    return fused.numpy(), sim.numpy()



def save_full_prototypes(fused_prototypes, sim_matrix, save_path):
    data = {
        "fused_prototypes": fused_prototypes,
        "sim_matrix": sim_matrix
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✔ Saved to {save_path}")



if __name__ == "__main__":
    prototypes_dir = "./TCGA-NSCLC/slide_prototypes"
    json_path = "./text/nsclc_coarse.json"
    bert_path = "./Text_encoder"
    save_path = "./TCGA-NSCLC/final_prototypes.pkl"
    split_csv = "./Datasets/TCGA-NSCLC/split.csv"

    df_split = pd.read_csv(split_csv)
    test_slide_ids = set(df_split["test"].dropna().astype(str).tolist())
    print("Test slides:", len(test_slide_ids))

    # 1) load slide prototypes
    all_prototypes, _ = load_all_slide_prototypes(
        prototypes_dir,
        test_slide_ids=test_slide_ids
    )

    # 2) dataset clustering
    dataset_prototypes = cluster_dataset_prototypes(
        all_prototypes,
        n_clusters=15
    )

    # 3) text features
    text_features = encode_text_features(
        json_path,
        bert_path,
        device="cpu"
    )

    # 4) fusion + sim
    fused_prototypes, sim_matrix = fuse_prototypes_with_text_concat(
        dataset_prototypes,
        text_features
    )

    # 5) save
    save_full_prototypes(
        fused_prototypes,
        sim_matrix,
        save_path
    )
