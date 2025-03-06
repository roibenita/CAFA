import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd


csv_path = "ANONYMIZED"
column_names = ["file", "number", "caption", "split"]
df = pd.read_csv(csv_path, names=column_names, header=None)
default_captions_dict = df.to_dict(orient="records")
file_to_default_captions = {
    f"{item['file']}_{item['number']}": item["caption"]
    for item in default_captions_dict
}

embeddings_folder = "ANONYMIZED"


def get_custom_metadata(info, audio):
    relpath = info["relpath"]
    filename = Path(relpath).stem
    key = filename.split("_train")[0]
    prompt = file_to_default_captions[key]
    embedding_path = os.path.join(embeddings_folder, f"{filename}_16khz25fps.npy")
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
