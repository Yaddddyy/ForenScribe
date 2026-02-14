import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

print("Script file:", __file__)
print("Working directory:", os.getcwd())
print("Files here:", os.listdir())
print("MPS available:", torch.backends.mps.is_available())
print("=======================================")

token = "HF_TOKEN"
model_id = "google/medgemma-4b-it"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=token
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token,
    torch_dtype=torch.float32,
    device_map=None
)

device = torch.device("mps")
model.to(device)


model.config.use_cache = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.train()

model.print_trainable_parameters()

dataset_path = "merged_forensic_dataset.jsonl"
print("Using dataset path:", dataset_path)

dataset = load_dataset("json", data_files=dataset_path, split="train")
print("Dataset columns:", dataset.column_names)

def tokenize_function(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        user_input = examples["input"][i]
        response = examples["output"][i]

        full_text = f"Instruction: {instruction}\nInput: {user_input}\nResponse: {response}"
        texts.append(full_text)

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=192
    )

    tokenized["labels"] = tokenized["input_ids"]

    if "token_type_ids" not in tokenized:
        tokenized["token_type_ids"] = [
            [0] * len(ids) for ids in tokenized["input_ids"]
        ]

    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

training_args = TrainingArguments(
    output_dir="./medgemma-forensic",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
    fp16=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./medgemma-forensic-lora")
tokenizer.save_pretrained("./medgemma-forensic-lora")

