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
    output_dir = Path(
        f"asym_cfg_ablation/{timestamp}{suffix_str}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir)


def load_data(
    vid_paths_list: List[str],
    embed_dir: str,
    captions_json_path: str,
    rank: int,
    world_size: int,
    captions: str,
    vgg_csv_path: str,
    all_vides: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset entries, distributed across GPUs.

    :param rank: Current GPU rank
    :param world_size: Total number of GPUs
    """
    # Load captions as before
    if captions == "default":
        column_names = ["file", "number", "caption", "split"]
        df = pd.read_csv(vgg_csv_path, names=column_names, header=None)
        captions_dict = df.to_dict(orient="records")
        file_to_captions = {item["file"]: item["caption"] for item in captions_dict}
    elif captions == "generated":
        with open(captions_json_path, "r") as f:
            captions_dict = json.load(f)
            file_to_captions = {
                item["file"]: item["generated_description"] for item in captions_dict
            }

    embed_paths, valid_vid_paths = [], []

    # Split video paths across GPUs
    if all_vides:
        all_embeds = list(Path(embed_dir).glob("*.npy"))

        files_already_done = set(
            Path(
                "ANONYMIZED"
            ).glob("*.wav")
        )
        files_already_done = [f.stem.split("_iter=1")[0] for f in files_already_done]
        all_embeds = [e for e in all_embeds if e.stem not in files_already_done]

        all_embeds = all_embeds[rank::world_size]
        # Distribute embed_paths across GPUs
        # local_embed_paths = embed_paths[rank::world_size]
        # valid_vid_paths = [p.stem + ".mp4" for p in local_embed_paths]

        for embed_path in all_embeds:
            vid_path = embed_path.stem + ".mp4"
            # if os.path.exists(vid_path):
            embed_paths.append(embed_path)
            valid_vid_paths.append(vid_path)
    else:
        # Distribute vid_paths across GPUs
        local_vid_paths = vid_paths_list[rank::world_size]
        print(f"GPU {rank} - number of local_vid_paths: {len(local_vid_paths)}")

        for vid_path in local_vid_paths:
            embed_path = os.path.join(
                embed_dir, os.path.basename(vid_path).replace(".mp4", ".npy")
            )
            if os.path.exists(embed_path):
                embed_paths.append(embed_path)
                valid_vid_paths.append(vid_path)
            else:
                logging.info(f"File not found (skipping): {embed_path}")

    logging.info(f"GPU {rank} processing {len(valid_vid_paths)} videos")

    data = []
    count_errs = 0
    for i, vid_path in enumerate(valid_vid_paths):
        clean_path = (
            vid_path.replace("_16khz25fps", "")
            if "_16khz25fps" in vid_path
            else vid_path
        )
        try:
            if captions == "default":
                prompt = file_to_captions[os.path.basename(clean_path)[:11]]
            elif captions == "generated":
                prompt = file_to_captions[os.path.basename(clean_path)]

            data.append(
                {
                    "filename": vid_path,
                    "embed_path": embed_paths[i],
                    "prompt": prompt,
                }
            )
        except KeyError:
            count_errs += 1
            logging.warning(f"KeyError for {vid_path} - no matching caption found.")

    logging.info(
        f"GPU {rank} - KeyError count: {count_errs} out of {len(valid_vid_paths)}"
    )
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
                "seconds_start": 0,
                "seconds_total": 10,  # 10-second target
            }
        )

    negative_conditioning = None

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


def main_worker(rank: int, world_size: int, config: dict, suffix: str):
    """
    Main worker function for each GPU.

    :param rank: Current GPU rank
    :param world_size: Total number of GPUs
    :param config: Configuration dictionary
    :param suffix: Output directory suffix
    """
    try:
        # Initialize distributed setup
        setup_distributed(rank, world_size)

        start_time = time.time()
        output_dir = prepare_output_directory(
            config["model_path"], config["steps"], config["captions"], suffix
        )

        # Set up logging for this GPU in a separate log directory
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"generation_gpu_{rank}.log"
        logging.basicConfig(
            level=logging.INFO,
            format=f"GPU{rank} - %(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )

        # Read video paths
        with open(config["vid_paths_list_path"], "r") as f:
            vid_paths_list = [line.strip() for line in f.readlines()]
        print(f"GPU {rank} - number of all vid_paths: {len(vid_paths_list)}")
        # Load data for this GPU
        data = load_data(
            vid_paths_list[: int(len(vid_paths_list) * config["data_portion"])],
            config["embed_dir"],
            config["captions_json_path"],
            rank,
            world_size,
            config["captions"],
            config["vgg_csv_path"],
            config["all_vides"],
        )

        logging.info(f"GPU {rank}: Loaded {len(data)} datapoints")

        # Load model on this GPU
        model, model_config = load_model(
            config["config_path"], config["model_path"], rank
        )

        # Calculate total generations for this GPU
        total_generations = len(config["cfgs"]) * config["num_generations"] * len(data)

        # Processing loop with progress bar
        with tqdm(
            total=total_generations,
            desc=f"GPU {rank} generating samples",
            position=rank,
        ) as pbar:
            for gen_idx in range(config["num_generations"]):
                # Use different seeds for each GPU to ensure diversity
                # seed = random.randint(0, 100000) + rank
                seed = 42
                seed_everything(seed)
                logging.info(f"GPU {rank} - seed: {seed}")
                for cfg in config["cfgs"]:
                    for batch_datapoints in chunk_list(data, config["batch_size"]):
                        try:
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
                                device=rank,  # Use GPU rank as device
                            )

                            # Save outputs
                            for dp, out_tensor in zip(batch_datapoints, outputs):
                                # output_name = f"{Path(dp['filename']).stem}_iter={gen_idx+1}_{Path(dp['filename']).stem}.wav"
                                output_name = f"{Path(dp['filename']).stem}.wav"
                                output_path = (
                                    Path(output_dir)
                                    # / f"iter={gen_idx+1}"
                                    # / f"cfg={cfg}"
                                    / output_name
                                )
                                output_path.parent.mkdir(parents=True, exist_ok=True)

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
        required=True,
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
        "captions_json_path": "ANONYMIZED",
        "vgg_csv_path": "ANONYMIZED",
        "vid_paths_list_path": "ANONYMIZED",
        "batch_size": 128,
        "data_portion": 1,
        "captions": "default",
        "all_vides": False,
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
