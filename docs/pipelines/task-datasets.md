# Task Datasets

Task datasets are the first reusable experiment substrate. They store task documents,
then run those documents through a causal language model while marking which tokens are
part of the behavior we care about.

## Dataset Contract

A task dataset is a versioned JSON object:

```json
{
  "schema_version": 1,
  "description": "Multiple-choice arithmetic prompts.",
  "documents": [
    {
      "text": "Question: 2+2?\nA. 3\nB. 4\nRESPONSE: B"
    },
    {
      "text": "Alice gave the book to Bob. The indirect object is Bob",
      "behavior_token_indices": [9]
    }
  ]
}
```

The raw JSON shape is simple:

- `documents` is non-empty.
- each document `text` is non-empty.
- `behavior_token_indices`, when present, is a non-empty list of unique non-negative
  token indices.

When `behavior_token_indices` is omitted, tokenization defaults the behavior mask to the
final token in the document. When it is present, indices refer to positions in the full
tokenized document produced by the chosen Hugging Face tokenizer, so those indices are
validated during tokenization.

## Tokenization

`tokenize_task_dataset(dataset, tokenizer)` tokenizes the whole document once with a
`transformers.PreTrainedTokenizerBase`. It does not separately tokenize an answer suffix,
because suffix tokenization can differ from tokenization inside the full prompt.

The tokenized result stores:

- `input_ids`: `Int64[torch.Tensor, "tokens"]`
- `behavior_mask`: `Bool[torch.Tensor, "tokens"]`

Behavior token index `0` is rejected for causal LM loss, because token `i` is scored from
the model logits at position `i - 1`.

## Model Execution

`run_task_dataset(dataset, model, tokenizer)` accepts a raw Hugging Face
`PreTrainedModel` and `PreTrainedTokenizerBase`. It tokenizes the dataset, pads documents,
runs the model, and returns:

- `input_ids`: `Int64[torch.Tensor, "batch tokens"]`
- `attention_mask`: `Bool[torch.Tensor, "batch tokens"]`
- `behavior_mask`: `Bool[torch.Tensor, "batch tokens"]`
- `logits`: `Float[torch.Tensor, "batch tokens vocab"]`
- `per_token_loss`: `Float[torch.Tensor, "batch tokens"]`

`TaskRun.mean_behavior_loss` averages only the selected behavior-token losses. It is a
measurement helper, not an equivalence or simplification metric by itself.

## Hugging Face Loader

The Hugging Face loader starts with `Qwen/Qwen3.5-0.8B` as the default local model id:

```python
from neural_program_simplification import (
    load_huggingface_causal_lm,
    load_task_dataset,
    run_task_dataset,
)

dataset = load_task_dataset("datasets/arithmetic-mcq-dev.json")
loaded = load_huggingface_causal_lm(device="cuda")
task_run = run_task_dataset(dataset, loaded.model, loaded.tokenizer)
```

The loader is intentionally thin: it returns the raw Hugging Face model and tokenizer in
a small dataclass, and model execution works directly against Hugging Face base classes.
