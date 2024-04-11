from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# prompts
prompts = [
    "What is the name of the bridge?",
    "What is the IP address of the bridge?",
    "What is the subnet mask of the bridge?",
    "What is the default gateway of the bridge?",
]

def generate_response(prompt):
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate text based on prompt
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    # Decode and return generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

for prompt in prompts:
    print(prompt)
    print(generate_response(prompt))
