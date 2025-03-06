import os
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd


json_path = "/home/naomi/mickey/stable_audio_ControlNet/as_final.json"
with open(json_path, "r") as f:
    data = json.load(f)["data"]

file_to_default_captions = {item["id"][1:12]: item["caption"] for item in data}

embeddings_folder = "/home/naomi/mickey/data/audioset_sl_train/avclip_embeds"


def get_custom_metadata(info, audio):
    relpath = info["relpath"]
    filename = Path(relpath).stem
    key = filename[:11]
    prompt = file_to_default_captions[key]
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
