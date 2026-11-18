"""Select a top-scoring subset, train CLIP (RN50) from scratch, and evaluate ImageNet zero-shot."""

import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import torchvision

from tqdm import tqdm

from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW

from dataloader.dataset import CLIP_CC3M_dataset_hugging, CLIP_CC12M_dataset_hugging
from dataloader.data_loaders import get_dataloader

from model.model import CLIP

from utils.simple_tokenizer import SimpleTokenizer
from utils.custom_schedulers import get_cosine_schedule_with_warmup
from utils import set_seed, mkdir, setup_logger, load_config_file

from zeroshot_eval import getWordnetId2ClassName, evaluate, eval_transform


TRAINER_CONFIG_PATH = "trainer/train_config.yaml"
MODEL_CONFIG_PATH = "model/model_config.yaml"


def get_model_state_dict(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def get_model_for_eval(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def load_arrow_dataset(dataset_name, data_root):
    if dataset_name == "cc3m":
        num_shards = 521
        prefix = "cc3m-wds-train"
        arrow_files = [
            os.path.join(data_root, f"{prefix}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.arrow")
            for i in range(num_shards)
        ]

    elif dataset_name == "cc12m":
        num_shards = 2224
        prefix = "webdataset-train"
        arrow_files = [
            os.path.join(data_root, f"{prefix}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.arrow")
            for i in range(num_shards)
        ]

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    missing = [f for f in arrow_files if not os.path.exists(f)]
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing arrow file: {missing[0]}")

    datasets = [Dataset.from_file(file) for file in arrow_files]
    train_dataset = concatenate_datasets(datasets)

    print("length of trainset:", len(train_dataset))
    return train_dataset


def select_subset(train_dataset, args):
    dataset_size = len(train_dataset)
    subset_size = int(args.subset_ratio * dataset_size)

    if args.subset_path is not None:
        print(f"Loading subset index from: {args.subset_path}")
        index = np.load(args.subset_path)
        index = index[:subset_size]

    elif args.score_name == "random":
        print("Using random subset.")
        rng = np.random.default_rng(args.seed)
        index = rng.permutation(dataset_size)[:subset_size]

    else:
        if args.score_path is None:
            raise ValueError("--score_path is required unless score_name is random or subset_path is given.")

        print(f"Loading score from: {args.score_path}")
        score = np.load(args.score_path)

        if len(score) != dataset_size:
            raise ValueError(
                f"Score length mismatch: len(score)={len(score)}, dataset_size={dataset_size}"
            )

        if args.score_order == "desc":
            ranking = (-score).argsort()
        elif args.score_order == "asc":
            ranking = score.argsort()
        else:
            raise ValueError(f"Unknown score_order: {args.score_order}")

        index = ranking[:subset_size]

    index = [int(i) for i in index]

    print("subset_ratio:", args.subset_ratio)
    print("subset_size:", len(index))

    return torch.utils.data.Subset(train_dataset, index)


def train_eval(config, train_dataset, model, subset_ratio, eval_dataloader, dataset_classes, description, logger, args):
    device = torch.device(config.device)

    print("Number of GPUs:", config.n_gpu)

    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = get_dataloader(config, train_dataset, is_train=True)

    # Fixed training budget: a subset with ratio r is trained for ~1/r times as many epochs,
    # so the total number of optimization steps matches full-data training.
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    t_total = int(t_total / subset_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.params.lr,
        eps=config.optimizer.params.eps,
        weight_decay=config.optimizer.params.weight_decay,
    )

    # Linear warmup for 20% of the steps, then cosine decay.
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=t_total,
    )

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)
    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size = %d",
        config.train_batch_size * config.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    global_step, global_loss = 0, 0.0
    model = model.to(device)
    model.train()
    model.zero_grad()

    for epoch in range(int(config.num_train_epochs)):
        model = model.to(device)
        model.train()

        train_iter = iter(train_dataloader)

        for step in tqdm(range(int(len(train_dataloader) / subset_ratio)), desc=f"Training epoch {epoch}"):
            try:
                input_images, input_texts = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                input_images, input_texts = next(train_iter)

            input_images = input_images.to(device, non_blocking=True)
            input_texts = input_texts.to(device, non_blocking=True)

            image_features, text_features = model(input_images, input_texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if config.n_gpu == 1:
                logit_scale = model.logit_scale.exp()
            elif config.n_gpu > 1:
                logit_scale = model.module.logit_scale.exp()
            else:
                logit_scale = model.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_per_text, labels)
            loss = (image_loss + text_loss) / 2

            if config.n_gpu > 1:
                loss = loss.mean()

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1

                optimizer.step()

                # CLIP logit scale clamp: log(100) = 4.6052
                if config.n_gpu == 1:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
                elif config.n_gpu > 1:
                    model.module.logit_scale.data = torch.clamp(model.module.logit_scale.data, 0, 4.6052)
                else:
                    model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

                if scheduler:
                    scheduler.step()

                model.zero_grad()

        logger.info(
            "Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(
                epoch,
                global_step,
                optimizer.param_groups[0]["lr"],
                loss.item(),
                global_loss / max(1, global_step),
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{description}.pth")

    torch.save(
        {"model_state_dict": get_model_state_dict(model)},
        checkpoint_path,
    )

    print(f"Saved checkpoint to: {checkpoint_path}")

    eval_model = get_model_for_eval(model)
    eval_model = eval_model.to(device)
    eval_model.eval()

    WordnetId2ClassName = getWordnetId2ClassName(args.wordnet_path)
    tokenizer = SimpleTokenizer()

    dataset_type = "imagenet"
    top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy = evaluate(
        eval_model,
        eval_dataloader,
        dataset_classes,
        WordnetId2ClassName,
        tokenizer,
        device,
        dataset_type,
    )

    print("top_1_accuracy, top_5_accuracy:", round(top_1_accuracy, 2), round(top_5_accuracy, 2))

    result_dir = os.path.dirname(args.result_path)
    if result_dir != "":
        os.makedirs(result_dir, exist_ok=True)

    data = (
        f"{description} "
        f"top1 {round(top_1_accuracy, 2)} "
        f"top5 {round(top_5_accuracy, 2)} "
        f"seed_{args.seed}\n"
    )

    with open(args.result_path, "a") as f:
        f.write(data)

    print(f"Saved result to: {args.result_path}")

    return top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="cc3m", type=str, choices=["cc3m", "cc12m"])
    parser.add_argument("--data_root", type=str, required=True, help="Directory containing arrow shards.")

    parser.add_argument("--score_name", default="random", type=str)
    parser.add_argument("--score_path", default=None, type=str)
    parser.add_argument("--subset_path", default=None, type=str)
    parser.add_argument("--subset_ratio", default=1.0, type=float)
    parser.add_argument("--score_order", default="desc", type=str, choices=["desc", "asc"])

    parser.add_argument("--seed", default=11, type=int)
    parser.add_argument("--num_train_epochs", default=None, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=None, type=int)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)

    parser.add_argument("--imagenet_val", type=str, required=True)
    parser.add_argument("--wordnet_path", default="zero_shot/WordNetId2ClassName.txt", type=str)
    parser.add_argument("--eval_batch_size", default=256, type=int)

    parser.add_argument("--output_dir", default="outputs/saved_model", type=str)
    parser.add_argument("--result_path", default="outputs/results.txt", type=str)
    parser.add_argument("--log_dir", default="outputs/logs", type=str)

    args = parser.parse_args()

    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)

    config = train_config

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()

    if args.num_train_epochs is not None:
        config.num_train_epochs = args.num_train_epochs

    if args.per_gpu_train_batch_size is not None:
        config.per_gpu_train_batch_size = args.per_gpu_train_batch_size

    global_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    if global_batch_size != 1024:
        print(
            f"[Warning] Global batch size is {global_batch_size} "
            f"({config.per_gpu_train_batch_size} per GPU x {max(1, config.n_gpu)} GPUs). "
            "The paper uses 1024."
        )

    if args.learning_rate is not None:
        config.optimizer.params.lr = args.learning_rate

    if args.weight_decay is not None:
        config.optimizer.params.weight_decay = args.weight_decay

    mkdir(path=args.output_dir)
    mkdir(path=args.log_dir)

    logger = setup_logger(
        "CLIP_SUBSET_TRAIN",
        args.log_dir,
        0,
        filename="training_logs.txt",
    )

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

    seed = args.seed
    print("seed:", seed)
    set_seed(seed=seed, n_gpu=config.n_gpu)

    tokenizer = SimpleTokenizer()

    model_params = dict(model_config.RN50)
    model_params["vision_layers"] = tuple(model_params["vision_layers"])
    model_params["vision_patch_size"] = None
    model = CLIP(**model_params)

    logger.info(f"Training/evaluation parameters {train_config}")

    print("\nDataset name:", args.dataset_name)
    train_dataset = load_arrow_dataset(args.dataset_name, args.data_root)

    description = f"{args.dataset_name}_{args.score_name}_{args.subset_ratio}_seed{seed}"
    print("description:", description)

    train_dataset = select_subset(train_dataset, args)

    if args.dataset_name == "cc12m":
        train_dataset = CLIP_CC12M_dataset_hugging(config, train_dataset, tokenizer)
    else:
        train_dataset = CLIP_CC3M_dataset_hugging(config, train_dataset, tokenizer)

    imageDataset = torchvision.datasets.ImageFolder(
        root=args.imagenet_val,
        transform=eval_transform(224),
    )

    eval_dataloader = DataLoader(
        imageDataset,
        sampler=SequentialSampler(imageDataset),
        batch_size=args.eval_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    dataset_classes = imageDataset.classes
    print("number of dataset classes:", len(dataset_classes))

    top_1_accuracy, top_5_accuracy, class_wise_top_1_accuracy, class_wise_top_5_accuracy = train_eval(
        config,
        train_dataset,
        model,
        args.subset_ratio,
        eval_dataloader,
        dataset_classes,
        description,
        logger,
        args,
    )

    print("Final ImageNet top1/top5:", top_1_accuracy, top_5_accuracy)


if __name__ == "__main__":
    main()