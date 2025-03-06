import os
import torch
import numpy as np
import pandas as pd


train_csv_path = "/home/naomi/mickey/stable_audio_ControlNet/train (3).csv"
df = pd.read_csv(train_csv_path)
captions_dict = {
    f"{row['youtube_id']}_{row['audiocap_id']}": row["caption"]
    for _, row in df.iterrows()
}

embeddings_folder = "/home/naomi/mickey/data/audiocaps/train/avclip_embeds"


def get_custom_metadata(info, audio):
    relpath = info["relpath"]
    filename = relpath.split(".wav")[0]
    prompt = captions_dict[filename]
    embedding_path = os.path.join(embeddings_folder, f"{filename}.npy")
    avclip_embedding = np.load(embedding_path)
    avclip_embedding = torch.from_numpy(avclip_embedding)

    if len(avclip_embedding.shape) == 2:
        avclip_embedding = avclip_embedding.unsqueeze(0)

    assert len(avclip_embedding.shape) == 3
    if avclip_embedding.shape[1] > 216:
        avclip_embedding = avclip_embedding[:, :216, :]
    elif avclip_embedding.shape[1] < 216:
        avclip_embedding = torch.cat(
            [avclip_embedding, torch.zeros(1, 216 - avclip_embedding.shape[1], 768)],
            dim=1,
        )
    assert avclip_embedding.shape == (1, 216, 768)

    return {"prompt": prompt, "avclip_signal": avclip_embedding}
