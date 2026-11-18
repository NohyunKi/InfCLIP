import argparse
import io
from pathlib import Path

import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader, SequentialSampler


class ArrowImageTextDataset(TorchDataset):
    def __init__(self, hf_dataset, preprocess, tokenizer, image_key="jpg", text_key="txt"):
        self.dataset = hf_dataset
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_key = image_key
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, img_obj):
        if isinstance(img_obj, dict):
            if img_obj.get("bytes") is not None:
                img = Image.open(io.BytesIO(img_obj["bytes"]))
            elif img_obj.get("path") is not None:
                img = Image.open(img_obj["path"])
            else:
                raise ValueError("Image dict should contain either 'bytes' or 'path'.")
        else:
            img = img_obj

        return img.convert("RGB")

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = self._load_image(sample[self.image_key])
        text = sample[self.text_key]

        image = self.preprocess(image)
        text = self.tokenizer([text])[0]

        return image, text


def load_arrow_dataset(dataset_name, data_root):
    data_root = Path(data_root)

    if dataset_name == "cc3m":
        num_shards = 521
        prefix = "cc3m-wds-train"
        arrow_files = [
            data_root / f"{prefix}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.arrow"
            for i in range(num_shards)
        ]

    elif dataset_name == "cc12m":
        num_shards = 2224
        prefix = "webdataset-train"
        arrow_files = [
            data_root / f"{prefix}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.arrow"
            for i in range(num_shards)
        ]

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    missing = [str(p) for p in arrow_files if not p.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing arrow file: {missing[0]}")

    datasets = [Dataset.from_file(str(path)) for path in arrow_files]
    dataset = concatenate_datasets(datasets)

    print("length of dataset:", len(dataset))
    return dataset


def extract_feature(model, dataloader, device):
    image_features_list, text_features_list = [], []

    model.eval()

    with torch.no_grad():
        for input_images, input_texts in tqdm(dataloader, desc="Extracting features"):
            input_images = input_images.to(device, non_blocking=True)
            input_texts = input_texts.to(device, non_blocking=True)

            if input_texts.ndim == 3 and input_texts.shape[1] == 1:
                input_texts = input_texts.squeeze(1)

            image_output = model.encode_image(input_images)
            text_output = model.encode_text(input_texts)

            image_features_list.append(image_output.detach().cpu())
            text_features_list.append(text_output.detach().cpu())

    image_features = torch.cat(image_features_list, dim=0)
    text_features = torch.cat(text_features_list, dim=0)

    print(f"Extracted image features shape: {image_features.shape}")
    print(f"Extracted text features shape : {text_features.shape}")

    return {
        "image_outputs": image_features,
        "text_outputs": text_features,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, required=True, choices=["cc3m", "cc12m"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="datacomp_xl_s13b_b90k")

    parser.add_argument("--image_key", type=str, default="jpg")
    parser.add_argument("--text_key", type=str, default="txt")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("========== Feature extraction ==========")
    print(f"Dataset name : {args.dataset_name}")
    print(f"Data root    : {args.data_root}")
    print(f"Output path  : {args.output_path}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Device       : {device}")
    print(f"Model        : {args.model_name}")
    print(f"Pretrained   : {args.pretrained}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    model = model.to(device)
    model.eval()

    hf_dataset = load_arrow_dataset(args.dataset_name, args.data_root)

    dataset = ArrowImageTextDataset(
        hf_dataset=hf_dataset,
        preprocess=preprocess,
        tokenizer=tokenizer,
        image_key=args.image_key,
        text_key=args.text_key,
    )

    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    image_text_features = extract_feature(model, dataloader, device)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(image_text_features, output_path)

    print("\n[Saved]")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()