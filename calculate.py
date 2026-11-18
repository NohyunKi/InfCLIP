import argparse
from pathlib import Path

import numpy as np
import torch

from calculate_score import calculate_inf_softmax_ntk


def load_feature_dict(path):
    """
    Load feature dictionary.

    Expected keys:
        - image_outputs
        - text_outputs

    Supported formats:
        - .pt / .pth saved by torch.save
        - .npy saved as a Python dict by np.save
        - .npz with image_outputs and text_outputs arrays
    """
    path = str(path)

    if path.endswith(".pt") or path.endswith(".pth"):
        data = torch.load(path, map_location="cpu", weights_only=False)

    elif path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            data = data.item()

    elif path.endswith(".npz"):
        data_npz = np.load(path, allow_pickle=True)
        data = {
            "image_outputs": data_npz["image_outputs"],
            "text_outputs": data_npz["text_outputs"],
        }

    else:
        raise ValueError(f"Unsupported feature file format: {path}")

    if "image_outputs" not in data or "text_outputs" not in data:
        raise KeyError(
            f"Feature file must contain keys 'image_outputs' and 'text_outputs'. "
            f"Found keys: {list(data.keys())}"
        )

    image_outputs = data["image_outputs"]
    text_outputs = data["text_outputs"]

    if not torch.is_tensor(image_outputs):
        image_outputs = torch.tensor(image_outputs)
    if not torch.is_tensor(text_outputs):
        text_outputs = torch.tensor(text_outputs)

    return {
        "image_outputs": image_outputs.cpu(),
        "text_outputs": text_outputs.cpu(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute InfCLIP scores from pre-extracted train features and "
            "class-mean target features."
        )
    )

    parser.add_argument(
        "--train_feature_path",
        type=str,
        required=True,
        help="Path to train image-text feature file. Expected keys: image_outputs, text_outputs.",
    )
    parser.add_argument(
        "--target_feature_path",
        type=str,
        required=True,
        help=(
            "Path to class-mean target image-text feature file. "
            "Expected keys: image_outputs, text_outputs. "
            "For ImageNet, each tensor should typically have shape (1000, D)."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save InfCLIP scores as .npy.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for InfCLIP score computation.",
    )
    parser.add_argument(
        "--logit_scale",
        type=float,
        default=10.0,
        help="Temperature-rescaled logit scale used in InfCLIP.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device, e.g., cuda, cuda:0, or cpu.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("========== InfCLIP score computation ==========")
    print(f"Train feature path : {args.train_feature_path}")
    print(f"Target feature path: {args.target_feature_path}")
    print(f"Output path        : {args.output_path}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Logit scale        : {args.logit_scale}")
    print(f"Device             : {device}")

    train_features = load_feature_dict(args.train_feature_path)
    target_features = load_feature_dict(args.target_feature_path)

    print("\n[Loaded features]")
    print("Train image features :", tuple(train_features["image_outputs"].shape))
    print("Train text features  :", tuple(train_features["text_outputs"].shape))
    print("Target image features:", tuple(target_features["image_outputs"].shape))
    print("Target text features :", tuple(target_features["text_outputs"].shape))

    score = calculate_inf_softmax_ntk(
        train_features_outputs=train_features,
        target_features_outputs=target_features,
        batch_size=args.batch_size,
        logit_scale=args.logit_scale,
        device=device,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, score)

    print("\n[Saved]")
    print(f"InfCLIP score shape: {score.shape}")
    print(f"Output path        : {output_path}")


if __name__ == "__main__":
    main()