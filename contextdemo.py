import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#  user  context
user_input = "Please create a linux bridge br0 using eth1 and eth2."

# Prompt the model to find out the name of the bridge
prompt = "What is the name of the bridge?"
prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
input_ids = torch.cat([tokenizer.encode(user_input, return_tensors="pt"), prompt_ids], dim=-1)

# Generate text based on user input and prompt
output = model.generate(input_ids, max_length=200, num_return_sequences=1)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)

# Check if the bridge name is mentioned in the generated text
if "br0" in generated_text:
    print("\nAI mentioned the bridge name 'br0'. Proceeding to ask about the ports.")
    # Ask about the ports of the bridge
    prompt = "What are the ports of the bridge?"
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = torch.cat([input_ids, prompt_ids], dim=-1)
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated text (asking about ports):")
    print(generated)
else:
    print("\nAI did not mention the bridge name. Unable to proceed with asking about ports.")
