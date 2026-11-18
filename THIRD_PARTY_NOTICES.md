# Third-party notices

This repository includes code adapted from the following projects.

## OpenAI CLIP

- Source: https://github.com/openai/CLIP
- Used in: `model/model.py`, `utils/simple_tokenizer.py`, `data/bpe_simple_vocab_16e6.txt.gz`,
  ImageNet class names and prompt templates in `zero_shot/class_names_and_templates.py`
- License: MIT

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hugging Face Transformers

- Source: https://github.com/huggingface/transformers
- Used in: `utils/custom_schedulers.py` (cosine learning-rate schedules with warmup)
- License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

## CLIP training code base

<!-- TODO(authors): confirm the repository this training loop / logger / config layout was
     adapted from (e.g. a public "clip-training" repository), and add its URL and license here. -->
- Used in: `subset_run.py` (training loop structure), `utils/logger.py`, `trainer/train_config.yaml`
