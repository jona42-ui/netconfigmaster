from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Encode input context
input_context = "Show me an exmaple of nmstate confiugration profiles"
input_ids = tokenizer.encode(input_context, return_tensors='pt')

# Generate text
output = model.generate(input_ids, max_length=500, num_return_sequences=1)

# Decode and print the output
print(tokenizer.decode(output[0], skip_special_tokens=True))
