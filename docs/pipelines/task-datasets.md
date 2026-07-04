# Task Datasets

Task datasets are the first reusable experiment substrate. They store prompts where the
final token is the model's task answer, then score that held-out final token with a
causal language model.

## Dataset Contract

A task dataset is a versioned JSON object:

```json
{
  "schema_version": 1,
  "dataset_id": "arithmetic-mcq-dev",
  "description": "Multiple-choice arithmetic prompts.",
  "metadata": {"split": "dev"},
  "examples": [
    {
      "id": "mcq-0001",
      "prompt": "Question: 2+2?\nA. 3\nB. 4\nRESPONSE: B",
      "answer_text": " B",
      "metadata": {"kind": "multiple-choice"}
    }
  ]
}
```

The raw JSON contract is tokenizer-independent:

- `dataset_id` and every example `id` are non-empty.
- `examples` is non-empty.
- example ids are unique.
- `prompt` is non-empty and must end with `answer_text`.
- `answer_text` is non-empty.

The tokenizer-specific contract is checked separately with
`validate_task_dataset_tokenization(dataset, tokenizer)`:

- `answer_text` must encode to exactly one token with `add_special_tokens=False`.
- the encoded prompt's final token id must equal that answer token id.
- the prompt must contain at least one context token before the answer token.

This keeps stored task datasets portable while making the model/tokenizer identity an
explicit part of any scored result.

## Model Execution

`run_task_dataset(tokenized_dataset, model)` treats the final prompt token as the task
answer. It feeds each prefix, excluding that answer token, into a `NextTokenModel` and
records the model's next-token logits for the answer.

Each `TaskPrediction` records:

- the example id
- the answer token id
- the top predicted token id
- answer and predicted logits
- answer log probability
- answer rank, with rank `1` meaning no token has a larger logit

`TaskRun.top1_accuracy` is a convenience summary over the scored examples. It is not an
equivalence or simplification metric by itself.

## Hugging Face Loader

The optional Hugging Face adapter starts with `Qwen/Qwen3.5-0.8B` as the default local
model id:

```python
from neural_program_simplification import (
    load_huggingface_causal_lm,
    load_task_dataset,
    run_task_dataset,
    validate_task_dataset_tokenization,
)

dataset = load_task_dataset("datasets/arithmetic-mcq-dev.json")
bundle = load_huggingface_causal_lm(device="cuda")
tokenized = validate_task_dataset_tokenization(dataset, bundle.tokenizer)
task_run = run_task_dataset(tokenized, bundle)
```

Install the optional runtime before loading real models:

```bash
uv sync --extra models
```

The adapter is intentionally narrow. It only exposes the next-token scoring surface
needed by task datasets; later tracing work should add activation hooks without changing
the dataset contract.
