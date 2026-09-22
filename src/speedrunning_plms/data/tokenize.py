"""Tokenize protein sequence records into binary shards of complete documents."""

import argparse
import glob
import multiprocessing as mp
import os

import numpy as np

from collections.abc import Iterable, Mapping
from functools import partial
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import EsmTokenizer

from speedrunning_plms.data.bin_format import write_shard


def upload_folder_to_hf(
    folder_path: str | Path,
    repo_id: str | None,
    repo_type: str = "dataset",
    token: str | None = None,
) -> None:
    """Upload a shard folder, reporting failures without aborting preprocessing."""
    if repo_id is None:
        print(f"Skipping upload for {folder_path} - no repo_id specified")
        return

    try:
        from huggingface_hub import HfApi

        api = HfApi()

        print(f"Uploading folder {folder_path} to {repo_id}...")

        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                exist_ok=True
            )
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Repository might already exist: {e}")

        file_count = len([f for f in os.listdir(folder_path) if f.endswith('.bin')])
        print(f"Found {file_count} files to upload")

        # Try to use multi_commits for large uploads (if supported)
        try:
            if file_count > 100:  # Use multi-commit for large uploads
                print("Using multi-commit upload for large number of files...")
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    multi_commits=True,
                    multi_commits_verbose=True
                )
            else:
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
        except TypeError as e:
            if "multi_commits" in str(e):
                print("multi_commits not supported in this version of huggingface_hub, using standard upload...")
                api.upload_folder(
                    folder_path=folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token
                )
            else:
                raise e

        print(f"Successfully uploaded folder {folder_path} to {repo_id}")

    except Exception as e:
        print(f"Error uploading folder {folder_path}: {e}")


def write_datafile(filename: str | Path, toks: np.ndarray) -> None:
    """Write uint8 tokens of shape (n,) after the fixed int32 header."""
    print(f"\nwriting {len(toks):,} tokens to {filename}")
    write_shard(filename, toks)


def tokenize(doc: Mapping[str, str], tokenizer: EsmTokenizer, max_length: int) -> np.ndarray:
    token_ids = tokenizer.encode(
        doc["sequence"],
        add_special_tokens=True,
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    return np.array(token_ids, dtype=np.uint8)  # (sequence_length,)


def tokenize_fw(
    fw: Iterable[Mapping[str, str]],
    split: str = 'train',
    data_name: str = 'omgprot50',
    max_length: int = 1024,
    upload_repo: str | None = None,
    token: str | None = None,
    shard_size: int | None = None,
    data_cache_dir: str | Path | None = None,
) -> None:
    """Write complete documents to shards, reusing any existing split files."""

    if shard_size is None:
        shard_size = 10**8
    if data_cache_dir is None:
        data_cache_dir = os.path.join(os.getcwd(), "data", data_name)

    existing_files = glob.glob(os.path.join(data_cache_dir, f"{data_name}_{split}_*.bin"))

    if existing_files:
        print(f"Found {len(existing_files)} existing .bin files for {data_name}_{split}")
        print("Skipping tokenization and proceeding to upload...")

        if upload_repo:
            upload_folder_to_hf(data_cache_dir, upload_repo, token=token)
        else:
            print("No upload repository specified, files are ready locally")
        return

    print(f"No existing .bin files found for {data_name}_{split}, proceeding with tokenization...")

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    nprocs = max(1, (os.cpu_count() or 1) - 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        current_shard: list[np.ndarray] = []
        current_size = 0
        progress_bar = None
        tokenize_fn = partial(tokenize, tokenizer=tokenizer, max_length=max_length)

        for tokens in pool.imap(tokenize_fn, fw, chunksize=16):  # tokens: (sequence_length,)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

            # If adding this sequence would exceed shard size, write current shard and start new one
            if current_size + len(tokens) > shard_size and current_size > 0:
                all_tokens = np.concatenate(current_shard)  # (current_size,)
                filename = os.path.join(data_cache_dir, f"{data_name}_{split}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens)

                shard_index += 1
                current_shard = []
                current_size = 0
                progress_bar = None

            current_shard.append(tokens)  # append (sequence_length,)
            current_size += len(tokens)
            if progress_bar:
                progress_bar.update(len(tokens))

        if current_size > 0:
            all_tokens = np.concatenate(current_shard)  # (current_size,)
            filename = os.path.join(data_cache_dir, f"{data_name}_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens)

    if upload_repo:
        upload_folder_to_hf(data_cache_dir, upload_repo, token=token)


parser = argparse.ArgumentParser(description="OMGprot50 dataset preprocessing")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-m", "--max_length", type=int, default=1024, help="Maximum sequence length")
parser.add_argument("-d", "--data_name", type=str, default="omg_prot50", help="Name of the dataset")
parser.add_argument("-r", "--upload_repo", type=str, default=None, help="Hugging Face repository ID to upload to (e.g., 'username/repo_name')")
parser.add_argument("-t", "--hf_token", type=str, default=None, help="Hugging Face token for authentication (or set token environment variable)")


def main() -> None:
    args = parser.parse_args()
    data_name = args.data_name

    token = args.hf_token or os.environ.get("token")
    if args.upload_repo and not token:
        print("Warning: Upload repository specified but no HF token provided. Set --hf_token or token environment variable.")

    data_cache_dir = os.path.join(os.getcwd(), "data", data_name)
    os.makedirs(data_cache_dir, exist_ok=True)

    train_fw = load_dataset(f"Synthyra/{data_name}", split="train")
    valid_fw = load_dataset(f"Synthyra/{data_name}", split="valid")
    test_fw = load_dataset(f"Synthyra/{data_name}", split="test")
    tokenize_fw(valid_fw, split='valid', data_name=data_name, max_length=args.max_length, upload_repo=args.upload_repo, token=token, shard_size=args.shard_size, data_cache_dir=data_cache_dir)
    tokenize_fw(test_fw, split='test', data_name=data_name, max_length=args.max_length, upload_repo=args.upload_repo, token=token, shard_size=args.shard_size, data_cache_dir=data_cache_dir)
    tokenize_fw(train_fw, split='train', data_name=data_name, max_length=100000, upload_repo=args.upload_repo, token=token, shard_size=args.shard_size, data_cache_dir=data_cache_dir)  # Keep the longer training sequence limit.


if __name__ == "__main__":
    main()
