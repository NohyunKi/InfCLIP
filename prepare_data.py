"""Download CC3M / CC12M from the Hugging Face Hub and lay out the Arrow shards
with the file names expected by extract_feature.py and subset_run.py.

The InfCLIP scores are indexed by the position of each sample in the concatenated
Arrow shards, so the shard order produced here must not be changed.

Example:
    python prepare_data.py --dataset_name cc3m --hf_repo pixparse/cc3m-wds \
        --cache_dir /path/to/hf_cache --output_dir /path/to/cc3m_arrow
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset

EXPECTED = {
    # dataset_name: (file prefix, number of shards used in the paper)
    "cc3m": ("cc3m-wds-train", 521),
    "cc12m": ("webdataset-train", 2224),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, choices=list(EXPECTED))
    parser.add_argument("--hf_repo", required=True, help="Hugging Face dataset repo id.")
    parser.add_argument("--cache_dir", required=True, help="Hugging Face datasets cache directory.")
    parser.add_argument("--output_dir", required=True, help="Directory that will hold the named Arrow shards.")
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    dataset = load_dataset(args.hf_repo, split="train", cache_dir=args.cache_dir, num_proc=args.num_proc)
    cache_files = [f["filename"] for f in dataset.cache_files]
    prefix, expected_shards = EXPECTED[args.dataset_name]

    print(f"Samples: {len(dataset)}")
    print(f"Arrow shards: {len(cache_files)} (paper setup: {expected_shards})")
    if len(cache_files) != expected_shards:
        print("[Warning] Shard count differs from the paper setup. Recompute features and scores "
              "for this copy instead of using released score files.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(cache_files)
    for i, src in enumerate(cache_files):
        dst = out / f"{prefix}-{i:05d}-of-{n:05d}.arrow"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)

    print(f"Linked {n} shards into {out}")


if __name__ == "__main__":
    main()
