import argparse
from pathlib import Path

import torch
import torchvision
import open_clip
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler


def get_wordnet_id_to_class_name(wordnet_path):
    mapping = {}

    with open(wordnet_path, "r") as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 3:
                continue

            wordnet_id = items[0]
            class_name = items[-1].replace("_", " ")
            mapping[wordnet_id] = class_name

    return mapping


IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


def extract_image_classmean(model, dataloader, num_classes, device):
    feature_sum = None
    class_count = torch.zeros(num_classes, dtype=torch.long)

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting image class-mean features"):
            images = images.to(device, non_blocking=True)
            labels = labels.cpu()

            image_outputs = model.encode_image(images).detach().cpu()

            if feature_sum is None:
                feature_dim = image_outputs.shape[-1]
                feature_sum = torch.zeros(num_classes, feature_dim, dtype=image_outputs.dtype)

            for i in range(image_outputs.shape[0]):
                label = int(labels[i])
                feature_sum[label] += image_outputs[i]
                class_count[label] += 1

    if torch.any(class_count == 0):
        missing = torch.where(class_count == 0)[0].tolist()
        raise ValueError(f"Missing classes in target dataset: {missing}")

    image_classmean = feature_sum / class_count[:, None]
    return image_classmean


def extract_text_classmean(model, tokenizer, classnames, device):
    text_features = []

    model.eval()

    with torch.no_grad():
        for classname in tqdm(classnames, desc="Extracting text class-mean features"):
            prompts = [template.format(classname) for template in IMAGENET_TEMPLATES]
            tokenized = tokenizer(prompts).to(device)

            text_outputs = model.encode_text(tokenized).detach()
            text_output = text_outputs.mean(dim=0)

            text_features.append(text_output.cpu())

    text_classmean = torch.stack(text_features, dim=0)
    return text_classmean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ImageNet class-mean target features using an OpenCLIP teacher."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to ImageNet validation directory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save target class-mean features.",
    )
    parser.add_argument(
        "--wordnet_path",
        type=str,
        default="zero_shot/WordNetId2ClassName.txt",
        help="Path to WordNetId2ClassName.txt.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="datacomp_xl_s13b_b90k",
        help="OpenCLIP pretrained checkpoint name.",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("========== Target feature extraction ==========")
    print(f"ImageNet root : {args.data_root}")
    print(f"Output path   : {args.output_path}")
    print(f"WordNet path  : {args.wordnet_path}")
    print(f"Model         : {args.model_name}")
    print(f"Pretrained    : {args.pretrained}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Device        : {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    model = model.to(device)
    model.eval()

    image_dataset = torchvision.datasets.ImageFolder(
        root=args.data_root,
        transform=preprocess,
    )

    dataset_classes = image_dataset.classes
    num_classes = len(dataset_classes)

    print(f"Number of classes: {num_classes}")
    print(f"First 5 class folder names: {dataset_classes[:5]}")

    if dataset_classes[0].startswith("n"):
        wordnet_id_to_class_name = get_wordnet_id_to_class_name(args.wordnet_path)
        classnames = [wordnet_id_to_class_name[c] for c in dataset_classes]
    else:
        raise ValueError(
            "This script currently expects ImageNet folders to be WordNet IDs "
            "such as n01440764. Please provide ImageNet val in ImageFolder format."
        )

    print(f"First 5 class names: {classnames[:5]}")

    dataloader = DataLoader(
        image_dataset,
        sampler=SequentialSampler(image_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    image_classmean = extract_image_classmean(
        model=model,
        dataloader=dataloader,
        num_classes=num_classes,
        device=device,
    )

    text_classmean = extract_text_classmean(
        model=model,
        tokenizer=tokenizer,
        classnames=classnames,
        device=device,
    )

    target_features = {
        "image_outputs": image_classmean,
        "text_outputs": text_classmean,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target_features, output_path)

    print("\n[Saved]")
    print(f"Image class-mean shape: {tuple(image_classmean.shape)}")
    print(f"Text class-mean shape : {tuple(text_classmean.shape)}")
    print(f"Output path           : {output_path}")


if __name__ == "__main__":
    main()