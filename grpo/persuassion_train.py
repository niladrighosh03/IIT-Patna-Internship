# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# The model ID for Llama 3.2 3B Instruct
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with the specified quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=quantization_config,
    device_map="auto", # Automatically map model layers to available devices
)

# %%
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# %%
sentence  = "Add the word 'Niladri' to the begging of the sentence hey bro"  
inputs = tokenizer(sentence, return_tensors="pt")

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id 
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# %%
import pandas as pd
from datasets import Dataset

# Load the CSV file
df = pd.read_csv('/DATA/rohan_kirti/niladri/grpo/all_conversations.csv')  # Replace 'your_file.csv' with your actual file name or path

# Extract only the 'utterance' column and drop rows with missing values
utterances = df['utterance'].dropna().tolist()

# Convert the utterances into the desired format
prompts_data = [{"prompt": utterance} for utterance in utterances]

# Convert the list of dictionaries to a Hugging Face Dataset object
train_dataset = Dataset.from_list(prompts_data)

print(train_dataset)

# %%
small_split = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = small_split["train"]
test_dataset = small_split["test"]


# %%
import re

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific strict format."""
    
    response_list = []
    for sentence in completions:
        sentence = "Add the word 'Niladri' to the beginning of the sentence and regenerate the response: " + sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_list.append(response)

    # Replace original list contents
    completions[:] = response_list
    
    
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, c.strip()) for c in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a loosely correct format."""
    
    
    response_list = []
    for sentence in completions:
        sentence = "Add the word 'Niladri' to the beginning of the sentence and regenerate the response: " + sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_list.append(response)

    # Replace original list contents
    completions[:] = response_list
    
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    matches = [re.search(pattern, c.strip(), re.DOTALL) for c in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text: str) -> float:
    """Internal utility to assign partial scores for XML-like formatting."""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward function giving partial score for structural XML-like format."""
    response_list = []
    for sentence in completions:
        sentence = "Add the word 'Niladri' to the beginning of the sentence and regenerate the response: " + sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_list.append(response)

    # Replace original list contents
    completions[:] = response_list
    return [count_xml(c.strip()) for c in completions]


def length_reward_func(completions, **kwargs):
    """
    A simple reward function that scores responses based on their length.

    Args:
        completions (list of str): A list of responses generated by the model.
        **kwargs: The trainer passes other arguments  here, which we ignore.

    Returns:
        list of float: A list of reward scores for each completion.
    """
    # The function returns a list of scores, one for each completion
    response_list=list()
    for sentence in completions:
        sentence  = "Add the word 'Niladri' to the begging of the sentence and regenerate the response"  + sentence
        inputs = tokenizer(sentence, return_tensors="pt")

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

        # Decode and print
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_list.append(response)
        
    completions[:] = response_list
    

    return [float(len(c)) for c in response_list]


def keyword_reward_func(completions, **kwargs):
    """
    Reward function that scores responses based on presence of specific persuasive/helpful keywords.
    
    Args:
        completions (list of str): A list of responses generated by the model.
        **kwargs: Additional arguments passed by the trainer (ignored here).
    
    Returns:
        list of float: A list of reward scores based on keyword matches.
    """
    
    response_list = []
    for sentence in completions:
        sentence = "Add the word 'Niladri' to the beginning of the sentence and regenerate the response: " + sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_list.append(response)

    # Replace original list contents
    completions[:] = response_list
    keywords = {"persuasion", "discount", "help", "offer", "support", "assist", "save", "deal"}
    
    rewards = []
    for c in completions:
        lowered = c.lower()
        hits = sum(1 for word in keywords if word in lowered)
        # Reward = base + 0.2 per keyword match, capped at 1.0
        reward = min(1.0, 0.2 * hits)
        rewards.append(reward)
    
    return rewards


def keyword_avoidance_reward_func(completions, **kwargs):
    response_list = []
    for sentence in completions:
        sentence = "Add the word 'Niladri' to the beginning of the sentence and regenerate the response: " + sentence
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id 
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_list.append(response)

    # Replace original list contents
    completions[:] = response_list
    bad_keywords = {"error", "unsure", "don't know", "not possible"}
    return [
        0.0 if any(bad in c.lower() for bad in bad_keywords) else 1.0
        for c in completions
    ]


# %%
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# GRPO training configuration
grpo_config = GRPOConfig(
    output_dir="/DATA/rohan_kirti/niladri/grpo/main",
    beta=0.1,  # The KL-divergence regularization coefficient
    max_prompt_length=256,
    max_completion_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=700,
    # max_steps=5,
    learning_rate=5e-5,
    logging_steps=35,
    report_to="tensorboard", # Set to "wandb" or "tensorboard" for experiment tracking
    num_generations=2,
)



# %%
# Initialize the trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    reward_funcs=[length_reward_func, keyword_avoidance_reward_func, strict_format_reward_func,
                  soft_format_reward_func, xmlcount_reward_func, keyword_reward_func], # Pass our reward function in a list
    peft_config=peft_config,
    
)

# Start the fine-tuning process
print("Starting GRPO fine-tuning...")
trainer.train()
print("Fine-tuning complete!")  



# %%
# Save the trained adapter model
trainer.save_model("/DATA/rohan_kirti/niladri/grpo/main/grpo_llama3.2_finetuned")


