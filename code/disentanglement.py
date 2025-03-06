from datetime import timedelta
import faulthandler

faulthandler.enable()
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Generator
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchaudio
from tqdm import tqdm
import logging
import gc
import pandas as pd
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model_local
import argparse


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(rank: int, world_size: int):
    """
    Enhanced distributed setup with NCCL-specific configurations
    """
    # rank = rank + 2
    try:
        # Set environment variables for NCCL
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        # NCCL-specific configurations
        os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging
        os.environ["NCCL_SOCKET_IFNAME"] = (
            "lo"  # Use loopback interface for local testing
        )
        os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand

        # Set cuda device before init
        torch.cuda.set_device(rank)

        # Initialize process group with NCCL
        logging.info(f"GPU {rank} - Initializing NCCL process group...")

        # NCCL-specific timeout settings
        timeout = timedelta(seconds=60)  # Increase timeout for NCCL init

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:29500",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        torch.cuda.synchronize()

        logging.info(f"GPU {rank} - NCCL process group initialized successfully")

        # Synchronize processes with explicit device mapping
        try:
            dist.barrier(device_ids=[rank])
            logging.info(f"GPU {rank} - NCCL barrier passed")
        except Exception as e:
            logging.error(f"GPU {rank} - NCCL barrier failed: {str(e)}")
            raise

    except Exception as e:
        logging.error(
            f"GPU {rank} - Failed to setup NCCL distributed training: {str(e)}"
        )
        raise


def cleanup_distributed():
    """Cleanup distributed training setup"""
    dist.destroy_process_group()


def load_model(config_path: str, model_path: str, rank: int):
    """
    Enhanced load_model function with detailed debugging
    """
    logging.info(f"GPU {rank} - Starting model load process...")

    # Check CUDA memory before loading
    if torch.cuda.is_available():
        logging.info(
            f"GPU {rank} - Initial CUDA memory: {torch.cuda.memory_allocated(rank) / 1e9:.2f} GB"
        )

    try:
        # Load model with extra logging
        logging.info(f"GPU {rank} - Loading model from {model_path}")
        model, model_config = get_pretrained_model_local(config_path, model_path)
        logging.info(
            f"GPU {rank} - Model loaded successfully. Model type: {type(model)}"
        )

        # Print model device location before moving
        logging.info(
            f"GPU {rank} - Model device before moving: {next(model.parameters()).device}"
        )

        # Move model to GPU with memory tracking
        logging.info(f"GPU {rank} - Moving model to GPU {rank}")
        model = model.to(rank)
        if torch.cuda.is_available():
            torch.cuda.synchronize(rank)
            logging.info(
                f"GPU {rank} - CUDA memory after model.to(): {torch.cuda.memory_allocated(rank) / 1e9:.2f} GB"
            )

        # Verify model is on correct device
        logging.info(
            f"GPU {rank} - Model device after moving: {next(model.parameters()).device}"
        )

        # Initialize DDP with extra checks
        logging.info(f"GPU {rank} - Initializing DDP...")
        for param in model.parameters():
            if not param.is_cuda:
                raise RuntimeError(
                    f"GPU {rank} - Found parameter on CPU before DDP wrap"
                )

        # state_dict = model.state_dict()
        # logging.info(f"GPU {rank} - Model state keys: {state_dict.keys()}")

        model = DDP(model, device_ids=[rank])
        logging.info(f"GPU {rank} - DDP initialization complete")

        # Final memory check
        if torch.cuda.is_available():
            torch.cuda.synchronize(rank)
            logging.info(
                f"GPU {rank} - Final CUDA memory: {torch.cuda.memory_allocated(rank) / 1e9:.2f} GB"
            )

        model.eval()
        model.unwraped = model
        return model, model_config

    except Exception as e:
        logging.error(f"GPU {rank} - Error in load_model: {str(e)}")
        logging.error(f"GPU {rank} - Error traceback:", exc_info=True)
        raise


def prepare_output_directory(
    model_path: str, steps: int, captions: str, suffix: str = ""
) -> str:
    """
    Create and return a timestamped output directory based on the model filename.
    All GPUs will save to the same directory.

    :param model_path: Path to the model checkpoint.
    :param steps: Number of diffusion steps.
    :param suffix: Optional suffix to append to the directory name.
    :return: String path of the newly created output directory.
    """
    timestamp = time.strftime("%m%d_%H%M")
    suffix_str = f"_{suffix}" if suffix else ""
    output_dir = Path(f"disentanglement/steps={steps}_{timestamp}{suffix_str}")
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir)


def load_data(
    embed_dir: str,
    rank: int,
    world_size: int,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset entries, distributed across GPUs.
    Each embedding will be processed once (captions handled in main loop).
    """
    # Get all embeddings from the directory and distribute across GPUs
    all_embeds = sorted(list(Path(embed_dir).glob("*.npy")))[rank::world_size]
    logging.info(f"GPU {rank} processing {len(all_embeds)} embeddings")

    data = []
    for embed_path in all_embeds:
        data.append(
            {
                "filename": embed_path.stem,
                "embed_path": str(embed_path),
            }
        )

    logging.info(f"GPU {rank} - Total datapoints: {len(data)}")
    return data


def chunk_list(lst: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    """
    Yield successive chunks of size batch_size from the list.

    :param lst: The list to chunk.
    :param batch_size: Size of each chunk.
    :return: Yields sublists of size batch_size.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def generate_audio_batch(
    model,
    datapoints: List[Dict[str, Any]],
    cfg: float,
    seed: int,
    sample_size: int,
    sample_rate: int,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    sampler_type: str,
    device: str,
) -> torch.Tensor:
    """
    Generate audio for a batch of datapoints. Each datapoint provides
    an AV-CLIP embedding and a text prompt.

    :param model: The loaded diffusion model.
    :param datapoints: A list of dicts with "av_clip_embedding" and "prompt".
    :param cfg: CFG (classifier-free guidance) scale.
    :param seed: A single random seed for all datapoints in the batch.
    :param sample_size: Desired audio sample size for generation.
    :param sample_rate: Sample rate for the output audio.
    :param steps: Diffusion steps.
    :param sigma_min: Minimum noise level for sampling.
    :param sigma_max: Maximum noise level for sampling.
    :param sampler_type: The diffusion sampler type (e.g., "dpmpp-3m-sde").
    :param device: The device to use ("cpu" or "cuda").
    :return: A torch tensor of shape [B, 1, T] containing generated audio,
             normalized, int16, and truncated to 10 seconds.
    """
    # Build the conditioning list
    conditioning_list = []
    for dp in datapoints:
        dp["av_clip_embedding"] = np.load(dp["embed_path"])
        dp["av_clip_embedding"] = torch.from_numpy(dp["av_clip_embedding"]).to(device)
        dp["av_clip_embedding"] = (
            dp["av_clip_embedding"].unsqueeze(0)
            if len(dp["av_clip_embedding"].shape) == 2
            else dp["av_clip_embedding"]
        )
        if len(dp["av_clip_embedding"].shape) != 3:
            logging.error(
                f"Embedding shape is not 3D: {dp['embed_path']} : {dp['av_clip_embedding'].shape}"
            )
            continue

        dim = dp["av_clip_embedding"].shape[1]
        if dim < 216:
            # it's [1, dim, 768], pad to [1, 216, 768] with zeroes
            logging.warning(
                f"Padding {dp['embed_path']} from shape {dp['av_clip_embedding'].shape} to [1, 216, 768] with zeroes"
            )

            dp["av_clip_embedding"] = torch.cat(
                [
                    dp["av_clip_embedding"],
                    torch.zeros(1, 216 - dim, 768).to(dp["av_clip_embedding"].device),
                ],
                dim=1,
            )
        elif dim > 216:
            # trim to [1, 216, 768]
            logging.warning(
                f"Trimming {dp['embed_path']} from shape {dp['av_clip_embedding'].shape} to [1, 216, 768]"
            )
            dp["av_clip_embedding"] = dp["av_clip_embedding"][:, :216, :]

        # randomly 1/250 of the datapoints, log the prompt
        if random.random() < 1 / 50:
            logging.info(
                f"Prompt: {dp['prompt']}, filename: {dp['filename']}, embedding: {dp['embed_path']}"
            )

        conditioning_list.append(
            {
                "avclip_signal": dp["av_clip_embedding"],
                "prompt": dp["prompt"],
                # "prompt": "gunshot",
                "seconds_start": 0,
                "seconds_total": 10,  # 10-second target
            }
        )

    negative_conditioning = [
        {
            "avclip_signal": condition["avclip_signal"],
            "prompt": dp["neg_prompt"],
            "seconds_start": 0,
            "seconds_total": 10,  # 10-second target
        }
        for condition in conditioning_list
    ]
    assert len(negative_conditioning) == len(conditioning_list)
    # negative_conditioning = None

    # Generate audio using the model
    output = generate_diffusion_cond(
        model.module if hasattr(model, "module") else model,
        cfg_scale=cfg,
        seed=seed,
        conditioning=conditioning_list,
        negative_conditioning=negative_conditioning,
        sample_size=sample_size,
        sample_rate=sample_rate,
        batch_size=len(datapoints),
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
    )

    # Normalize & clamp
    max_val_batch = torch.amax(torch.abs(output), dim=(1, 2), keepdim=True) + 1e-8
    output_batch_norm = output / max_val_batch
    output_batch_norm = output_batch_norm.clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    # Truncate or pad to 10 seconds exactly
    max_length = sample_rate * 10  # 10 seconds
    output_batch_norm = output_batch_norm[:, :, :max_length]

    return output_batch_norm

vgg_categories_csv = "ANONYMIZED"
# csv doesn't have column names, so we need to specify them
vgg_categories_df = pd.read_csv(
    vgg_categories_csv, names=["ytid", "start_time", "caption", "split"]
)

lookup_dict = {}
for _, row in vgg_categories_df.iterrows():
    key = (row["ytid"], row["start_time"])
    lookup_dict[key] = row["caption"]

def main_worker(rank: int, world_size: int, config: dict, suffix: str):
    """
    Main worker function for each GPU.
    """
    try:
        # Initialize distributed setup
        setup_distributed(rank, world_size)

        start_time = time.time()
        output_dir = prepare_output_directory(
            config["model_path"], config["steps"], "fixed-captions", suffix
        )

        # Set up logging
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"generation_gpu_{rank}.log"
        logging.basicConfig(
            level=logging.INFO,
            format=f"GPU{rank} - %(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )

        logging.info(f"Captions: {config['captions']}")
        # Load data for this GPU
        data = load_data(config["embed_dir"], rank, world_size)
        logging.info(f"GPU {rank}: Loaded {len(data)} embeddings")

        # Load model on this GPU
        model, model_config = load_model(
            config["config_path"], config["model_path"], rank
        )

        # Calculate total generations for this GPU
        total_generations = (
            len(config["captions"])
            * len(config["cfgs"])
            * config["num_generations"]
            * len(data)
        )

        # Processing loop with progress bar
        with tqdm(
            total=total_generations,
            desc=f"GPU {rank} generating samples",
            position=rank,
        ) as pbar:
            for caption in config["captions"]:
                for gen_idx in range(config["num_generations"]):
                    seed = 42
                    seed_everything(seed)
                    logging.info(
                        f"GPU {rank} - Processing caption: {caption}, seed: {seed}"
                    )

                    for cfg in config["cfgs"]:
                        for batch_datapoints in chunk_list(data, config["batch_size"]):
                            try:
                                # Add caption to each datapoint in batch
                                for dp in batch_datapoints:
                                    dp["prompt"] = caption
                                    dp["neg_prompt"] = lookup_dict[
                                        (
                                            dp["filename"][:11],
                                            int(dp["filename"][12:].split("_")[0]),
                                        )
                                    ]

                                generation_start = time.time()

                                # Generate audio
                                outputs = generate_audio_batch(
                                    model=model,
                                    datapoints=batch_datapoints,
                                    cfg=cfg,
                                    seed=seed,
                                    sample_size=model_config["sample_size"],
                                    sample_rate=model_config["sample_rate"],
                                    steps=config["steps"],
                                    sigma_min=config["sigma_min"],
                                    sigma_max=config["sigma_max"],
                                    sampler_type=config["sampler_type"],
                                    device=rank,
                                )

                                # Save outputs
                                for dp, out_tensor in zip(batch_datapoints, outputs):
                                    output_name = f"{dp['filename']}.wav"
                                    output_path = (
                                        Path(output_dir)
                                        / f"iter={gen_idx+1}"
                                        / f"cfg={cfg}"
                                        / caption
                                        / output_name
                                    )
                                    output_path.parent.mkdir(
                                        parents=True, exist_ok=True
                                    )

                                    torchaudio.save(
                                        output_path,
                                        out_tensor,
                                        model_config["sample_rate"],
                                        channels_first=True,
                                    )

                                generation_time = time.time() - generation_start
                                pbar.set_postfix(
                                    {"Generation Time": f"{generation_time:.2f}s"}
                                )
                                pbar.update(len(batch_datapoints))

                                # Memory management
                                if torch.cuda.is_available():
                                    del outputs
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    torch.cuda.synchronize()

                            except Exception as e:
                                logging.error(
                                    f"GPU {rank} - Error generating batch: {str(e)}"
                                )
                                pbar.update(len(batch_datapoints))
                                continue

        total_time = time.time() - start_time
        logging.info(f"GPU {rank} - Total processing time: {total_time:.2f} seconds")
        logging.info(
            f"GPU {rank} - Average time per generation: {total_time / total_generations:.2f} seconds"
        )

    except Exception as e:
        logging.error(f"GPU {rank} - Fatal error: {str(e)}")
    finally:
        cleanup_distributed()


def main():
    """
    Main entry point that spawns processes for each GPU.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix",
        nargs="?",
        default="",
        required=False,
        help="Optional suffix for the output directory",
    )
    args = parser.parse_args()

    if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
        logging.info(f"NCCL Version: {torch.cuda.nccl.version()}")

    # Configuration
    config = {
        "cfgs": [7],
        "num_generations": 1,
        "steps": 50,
        "sigma_min": 0.5,
        "sigma_max": 500.0,
        "sampler_type": "dpmpp-3m-sde",
        "config_path": "stable_audio_tools/csa/csa_model_config_controlnet_avclip.json",
        "model_path": "ANONYMIZED",
        "embed_dir": "ANONYMIZED",
        "batch_size": 16,
        "captions": [
            "striking bowling",
            "skateboarding",
            "playing tennis",
            "playing badminton",
            "people sneezing",
            "people eating apple",
            "people eating crisps",
            "lions roaring",
            "ice cracking",
            "hammering nails",
            "dog barking",
            "chopping wood",
        ],
    }

    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    # world_size = world_size - 2
    if world_size == 0:
        raise RuntimeError("No CUDA devices available")

    # world_size = 1
    logging.info(f"Starting distributed training with {world_size} GPUs")

    # Spawn processes for each GPU
    import torch.multiprocessing as mp

    try:
        mp.spawn(
            main_worker,
            args=(world_size, config, args.suffix),
            nprocs=world_size,
            join=True,
        )
    except Exception as e:
        logging.error(f"Error: {str(e)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        with open("error_log.txt", "w") as f:
            f.write(str(e))
        logging.info("Error message saved to error_log.txt")
