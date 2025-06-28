# LLM-CausalLanguageModelling

This repository provides a complete pipeline for **fine-tuning causal language models** such as DistilGPT2 on a subset of the [SQuAD dataset](https://huggingface.co/datasets/squad). It covers dataset preprocessing, training, and inference using Hugging Face's `transformers` and `datasets` libraries.

---

## Overview

Causal Language Modeling (CLM) is used to predict the next token in a sequence, enabling applications such as Text generation, Code completion (e.g., GitHub Copilot, CodeParrot) and Language understanding for autoregressive models like GPT-2 etc...

---

## Installation

Install required packages:

```bash
pip install transformers datasets evaluate
```

Or install the latest dev version of Transformers:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

---

## Dataset

This project uses a **5000-sample subset** of the [SQuAD](https://huggingface.co/datasets/squad) dataset:

```python
from datasets import load_dataset

squad = load_dataset("squad", split="train[:5000]")
squad = squad.train_test_split(test_size=0.2)
```

---

## Preprocessing

1. **Flatten nested fields** in the dataset  
2. **Tokenize** text using `DistilGPT2` tokenizer  
3. **Concatenate and chunk** sequences into fixed lengths for training  

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
```

---

## Training

Training is performed using Hugging Face’s `Trainer` API with `DataCollatorForLanguageModeling`:

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

training_args = TrainingArguments(
    output_dir="my_awesome_squad_clm-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
```

---

## Evaluation

```python
import math

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
```

---

## Inference

After training, you can generate text using your model with the Hugging Face pipeline or manually:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="my_awesome_squad_clm-model")
generator("Somatic hypermutation allows the immune system to")
```

Or manually:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("my_awesome_squad_clm-model")
model = AutoModelForCausalLM.from_pretrained("my_awesome_squad_clm-model")

inputs = tokenizer("Somatic hypermutation allows the immune system to", return_tensors="pt").input_ids
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

---

## Notes

- Uses Hugging Face 🤗 libraries for tokenizer/model/dataset handling.
- Easily extendable to other causal models like GPT-2, GPT-Neo, or LLaMA.
- Model is pushed to Hugging Face Hub for sharing.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [SQuAD Dataset](https://huggingface.co/datasets/squad)
