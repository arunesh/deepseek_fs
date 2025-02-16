
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from dotenv import load_dotenv

load_dotenv()

"""Load up `Llama 3.1 8B Instruct`, and set parameters"""

from unsloth import is_bfloat16_supported
import torch
max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# 1. Load the LoRA-trained model

# Replace with the actual path to your LoRA-trained model directory
lora_model_path = "path/to/your/lora/model"

model, tokenizer = FastLanguageModel.from_pretrained(
    lora_model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "test") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

test_dataset = get_gsm8k_questions()

def evaluate_model(model, tokenizer, dataset):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        prompt = sample['prompt']
        true_answer = sample['answer']

        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=786, pad_token_id=tokenizer.eos_token_id)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_xml_answer(generated_text)

        if predicted_answer == true_answer:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

accuracy = evaluate_model(model, tokenizer, test_dataset)
print(f"Accuracy: {accuracy}")
return accuracy
