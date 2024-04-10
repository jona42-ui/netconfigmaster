from transformers import GPT2LMHeadModel, GPT2Tokenizer
import yaml

def generate_yaml(input_text):
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate text based on the input
    output = model.generate(input_ids, max_length=500, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract bridge name and interface names from the generated text
    bridge_name = None
    interface_names = []

    # Split the generated text into lines and iterate through them
    for line in generated_text.split('\n'):
        # Check if the line contains information about a bridge
        if "linux bridge" in line.lower():
            bridge_name = line.split()[-1]  # Extract the last word as the bridge name
        # Check if the line contains information about interfaces
        elif "using" in line.lower():
            interface_names = line.split("using")[-1].strip().split(" and ")

    # Construct YAML configuration
    yaml_config = {
        "interfaces": [
            {
                "name": bridge_name,
                "type": "linux-bridge",
                "state": "up",
                "bridge": {
                    "ports": [{"name": interface} for interface in interface_names]
                }
            }
        ]
    }

    return yaml.dump(yaml_config)

if __name__ == "__main__":
    # Sample input for demo purposes
    input_text = "Please create a linux bridge br0 using eth1 and eth2."

    # Generate YAML configuration
    yaml_output = generate_yaml(input_text)

    # Print the YAML configuration
    print(yaml_output)
