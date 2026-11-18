"""ImageNet zero-shot evaluation for CLIP models trained with subset_run.py."""

import argparse
import os

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from dataloader.dataset import TRAIN_MEAN, TRAIN_STD
from model.model import CLIP
from utils import load_config_file, mkdir
from utils.simple_tokenizer import SimpleTokenizer
from zero_shot.class_names_and_templates import imagenet_classes, imagenet_templates

MODEL_CONFIG_PATH = "model/model_config.yaml"


def eval_transform(n_px=224):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(TRAIN_MEAN, TRAIN_STD),
    ])


def getWordnetId2ClassName(WordnetId2ClassName_filepath):
    """Map ImageNet WordNet IDs (e.g. n01440764) to class names (last field, '_' -> ' ')."""
    WordnetId2ClassName = {}
    with open(WordnetId2ClassName_filepath) as fp:
        for line in fp:
            items = line.split()
            if len(items) < 2:
                continue
            WordnetId2ClassName[items[0]] = " ".join(items[-1].split("_"))
    return WordnetId2ClassName


def tokenize(texts, tokenizer, context_length=77):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def zeroshot_classifier(model, classnames, templates, tokenizer, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, desc="Building zero-shot classifier"):
            texts = [template.format(classname) for template in templates]
            texts = tokenize(texts, tokenizer).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def get_imagenet_classnames(dataset_classes, WordnetId2ClassName):
    # Class folders named by WordNet ID (standard ImageNet val layout)
    if dataset_classes[0].startswith("n"):
        return [WordnetId2ClassName[c] for c in dataset_classes]
    # Class folders named by index (e.g. ImageNetV2)
    return [imagenet_classes[int(c)] for c in dataset_classes]


def evaluate(model, eval_dataloader, dataset_classes, WordnetId2ClassName, tokenizer, device,
             dataset_type="imagenet"):
    """Returns top-1 / top-5 accuracy (%) and class-wise accuracies."""
    if dataset_type != "imagenet":
        raise ValueError("Only ImageNet-style zero-shot evaluation is supported.")

    classnames = get_imagenet_classnames(dataset_classes, WordnetId2ClassName)

    top_1_correct, top_5_correct, total = 0, 0, 0
    class_top1 = {c: 0 for c in classnames}
    class_top5 = {c: 0 for c in classnames}
    class_total = {c: 0 for c in classnames}

    with torch.no_grad():
        zeroshot_weights = zeroshot_classifier(model, classnames, imagenet_templates, tokenizer, device)

        for images, labels in tqdm(eval_dataloader, desc="Zero-shot evaluation"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights          # (B, C)
            top5 = logits.topk(5, dim=-1).indices                        # (B, 5)

            hit1 = top5[:, 0] == labels
            hit5 = (top5 == labels[:, None]).any(dim=1)

            top_1_correct += hit1.sum().item()
            top_5_correct += hit5.sum().item()
            total += labels.numel()

            for lbl, h1, h5 in zip(labels.tolist(), hit1.tolist(), hit5.tolist()):
                name = classnames[lbl]
                class_total[name] += 1
                class_top1[name] += int(h1)
                class_top5[name] += int(h5)

    top_1_accuracy = 100 * top_1_correct / total
    top_5_accuracy = 100 * top_5_correct / total
    class_wise_top_1_accuracy = {c: 100 * class_top1[c] / class_total[c] for c in classnames if class_total[c] > 0}
    class_wise_top_5_accuracy = {c: 100 * class_top5[c] / class_total[c] for c in classnames if class_total[c] > 0}

    return top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy


def save_accuracies(path, top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy):
    with open(path, "w") as fp:
        fp.write(f"Top 1 accuracy = {round(top_1_accuracy, 2)} %\n")
        fp.write(f"Top 5 accuracy = {round(top_5_accuracy, 2)} %\n")
        fp.write("-----------------------------------------------------\n")
        fp.write("Class wise accuracies (in %)\n\n")
        fp.write("{:<10}{:<10}{:<20}\n".format("Top-1", "Top-5", "Class name"))
        for class_name in class_wise_top_1_accuracy:
            fp.write("{:<10}{:<10}{:<20}\n".format(
                round(class_wise_top_1_accuracy[class_name], 2),
                round(class_wise_top_5_accuracy[class_name], 2),
                class_name,
            ))


def build_model():
    model_config = load_config_file(MODEL_CONFIG_PATH)
    model_params = dict(model_config.RN50)
    model_params["vision_layers"] = tuple(model_params["vision_layers"])
    model_params["vision_patch_size"] = None
    return CLIP(**model_params)


def main():
    parser = argparse.ArgumentParser(description="ImageNet zero-shot evaluation of a trained checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="ImageNet val directory (ImageFolder layout).")
    parser.add_argument("--wordnet_path", type=str, default="zero_shot/WordNetId2ClassName.txt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/zero_shot_eval")
    args = parser.parse_args()

    mkdir(path=args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()

    dataset = torchvision.datasets.ImageFolder(root=args.data_dir, transform=eval_transform())
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True)

    results = evaluate(model, dataloader, dataset.classes, getWordnetId2ClassName(args.wordnet_path),
                       SimpleTokenizer(), device)
    top_1_accuracy, top_5_accuracy = results[0], results[1]
    print("top_1_accuracy, top_5_accuracy:", round(top_1_accuracy, 2), round(top_5_accuracy, 2))

    name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    out_path = os.path.join(args.output_dir, f"imagenet_{name}.txt")
    save_accuracies(out_path, *results)
    print("Class-wise accuracies saved to:", out_path)


if __name__ == "__main__":
    main()
