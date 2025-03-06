import os
import torch
import numpy as np
import pandas as pd


train_csv_path = "data/audiocaps_train.csv"
df = pd.read_csv(train_csv_path)
captions_dict = {
    f"{row['youtube_id']}_{row['audiocap_id']}": row["caption"]
    for _, row in df.iterrows()
}

videos_folder = "data/audiocaps/train/videos"
embeddings_folder = "data/audiocaps/train/synchformer_embeddings"

def get_custom_metadata(info, audio):
    relpath = info["relpath"]
    filename = relpath.split(".wav")[0]
    prompt = captions_dict[filename]
    embedding_path = os.path.join(embeddings_folder, f"{filename}.npy")
    avclip_embedding = np.load(embedding_path)
    avclip_embedding = torch.from_numpy(avclip_embedding)

    return {"prompt": prompt, "avclip_signal": avclip_embedding}
