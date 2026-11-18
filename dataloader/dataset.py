"""Image-text datasets for training CLIP on CC3M / CC12M (Hugging Face Arrow shards)."""

import io
import re

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Mean / std used for all CLIP models trained from scratch in the paper.
TRAIN_MEAN = (0.4225, 0.4012, 0.3659)
TRAIN_STD = (0.2681, 0.2635, 0.2763)


def _transform(n_px):
    """Resize + center crop (no augmentation), as used in all subset-training experiments."""
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize(TRAIN_MEAN, TRAIN_STD),
    ])


def _load_image(img_obj):
    """Decode an Arrow image cell (PIL image or {'bytes', 'path'} dict). Returns None if unreadable."""
    try:
        if isinstance(img_obj, dict):
            if img_obj.get("bytes") is not None:
                img = Image.open(io.BytesIO(img_obj["bytes"]))
            else:
                img = Image.open(img_obj["path"])
        else:
            img = img_obj
        return img.convert("RGB")
    except (OSError, SyntaxError, UnidentifiedImageError):
        return None


class CLIP_CC3M_dataset_hugging(Dataset):
    """Wraps a Hugging Face dataset with 'jpg' and 'txt' columns."""

    def __init__(self, config, dataset, text_tokenizer, context_length=77, input_resolution=224,
                 image_key="jpg", text_key="txt"):
        self.config = config
        self.dataset = dataset
        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length
        self.image_key = image_key
        self.text_key = text_key

    def _encode(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        # NOTE: captions longer than the context are truncated as in the paper experiments.
        tokens = tokens[:self.context_length]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def tokenize(self, text):
        try:
            return self._encode(text)
        except Exception:
            # Fall back to a sanitized caption if the BPE tokenizer fails on unusual characters.
            return self._encode(re.sub(r"[^a-zA-Z0-9 .,]", "", text))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = _load_image(sample[self.image_key])
        if img is None:
            # Skip corrupted images by moving to the next sample.
            return self.__getitem__((idx + 1) % len(self.dataset))

        text_input = self.tokenize(sample[self.text_key])
        img_input = self.transform(img)
        return img_input, text_input


class CLIP_CC12M_dataset_hugging(CLIP_CC3M_dataset_hugging):
    """CC12M uses the same format and processing as CC3M."""
