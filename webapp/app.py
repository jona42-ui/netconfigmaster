import os
import subprocess
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_file
import libnmstate
import yaml
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch

app = Flask(__name__)

# Path configurations
APP_DIRECTORY = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIRECTORY = APP_DIRECTORY.parent / 'model'
NETVISOR_PATH = APP_DIRECTORY.parent.parent / 'NetVisor' / 'target' / 'debug' / 'netvisor'

# Load the model and tokenizer
try:
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIRECTORY))
    tokenizer = GPT2Tokenizer.from_pretrained(str(MODEL_DIRECTORY))
    model.eval()  # Set to evaluation mode
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    tokenizer = None

def generate_nmstate_yaml(prompt):
    """Generate Nmstate YAML from natural language prompt"""
    try:
        # Prepare input
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_yaml = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Validate YAML
        yaml_dict = yaml.safe_load(generated_yaml)
        return yaml.dump(yaml_dict, default_flow_style=False)
    except Exception as e:
        raise ValueError(f"Error generating YAML: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get user input
        user_input = request.form['inputText']

        # Get current network state
        current_state = libnmstate.show()
        
        # Generate new state from natural language
        generated_yaml = generate_nmstate_yaml(user_input)
        
        # Parse the generated YAML
        desired_state = yaml.safe_load(generated_yaml)
        
        # Apply the network configuration
        libnmstate.apply(desired_state)
        
        # Get updated network state
        new_state = libnmstate.show()
        nmstate_output = yaml.dump(new_state, default_flow_style=False)

        return render_template('index.html', 
                             input_text=user_input, 
                             nmstate_output=nmstate_output,
                             success_message="Network configuration applied successfully")
    except Exception as e:
        return render_template('index.html', 
                             input_text=user_input,
                             error_message=f"Error: {str(e)}")

@app.route('/show_image', methods=['GET'])
def show_image():
    try:
        # Generate the image file
        output_path = APP_DIRECTORY / 'static' / 'output.png'
        netvisor_output = subprocess.run(
            [str(NETVISOR_PATH), 'show', '--file', str(output_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if netvisor_output.returncode == 0:
            return send_file(str(output_path), mimetype='image/png')
        else:
            return "Error generating image"
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if model is None:
        print("Warning: Model not loaded. NLP features will not work.")
    app.run(debug=True)
