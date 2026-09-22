import argparse
import os

from huggingface_hub import hf_hub_download


def get(fname: str, data_name: str) -> None:
    """Download one packed shard unless the local file already exists."""
    local_dir = os.path.join(os.getcwd(), "data", data_name)
    if not os.path.exists(os.path.join(local_dir, fname)):
        try:
            print(f"Downloading {fname} from Synthyra/{data_name}_packed")
            hf_hub_download(repo_id=f"Synthyra/{data_name}_packed", filename=fname, repo_type="dataset", local_dir=local_dir)
        except Exception as e:
            print(f"Error downloading {fname}: {e}")
    else:
        print(f"File {fname} already exists in {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data from huggingface")
    parser.add_argument("-d", "--data_name", type=str, default="uniref50", help="Name of the dataset, uniref50, omg_prot50, or og_prot90")
    parser.add_argument("-n", "--num_chunks", type=int, default=100, help="Number of chunks to download")
    # each chunk is 100M tokens
    args = parser.parse_args()
    get(f"{args.data_name}_valid_000000.bin", args.data_name)
    get(f"{args.data_name}_test_000000.bin", args.data_name)
    for i in range(0, args.num_chunks+1):
        get(f"{args.data_name}_train_{i:06d}.bin", args.data_name)


if __name__ == "__main__":
    main()
