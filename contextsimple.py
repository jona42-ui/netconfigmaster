from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define user input context
user_input = "Please create a linux bridge br0 using eth1 and eth2."

# Encode user input context
input_ids = tokenizer.encode(user_input, return_tensors="pt")

# Generate text based on user input
output = model.generate(input_ids, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)

# Check if the bridge name is mentioned in the generated text
if "br0" in generated_text:
    print("\nAI mentioned the bridge name 'br0'. Proceeding to ask about the ports.")
    # Ask about the ports of the bridge
    prompt = "What are the ports of the bridge?"
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Concatenate user input context and prompt
    input_ids = torch.cat([input_ids, prompt_ids], dim=-1)
    # Generate text based on concatenated input
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    # Decode and print generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated text (asking about ports):")
    print(generated_text)
else:
    print("\nAI did not mention the bridge name. Unable to proceed with asking about ports.")
