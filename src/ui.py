"""
This script provides a Gradio-based web interface for testing and evaluating the network
configuration model. It allows users to input natural language commands, view the generated
Nmstate YAML, validate the YAML syntax, and provide feedback on the model's performance.
"""

import argparse
import csv
import os
import re
import yaml
import gradio as gr
import torch
from transformers import pipeline
import libnmstate
from utils import load_model


def validate_yaml(yaml_str):
    """Validate if the generated YAML is valid and follows Nmstate schema"""
    try:
        # Parse YAML
        yaml_dict = yaml.safe_load(yaml_str)
        
        # Basic Nmstate schema validation
        if not isinstance(yaml_dict, dict):
            return False, "Generated YAML must be a dictionary"
            
        required_keys = {"interfaces", "routes", "dns-resolver", "route-rules"}
        if not any(key in yaml_dict for key in required_keys):
            return False, "Missing required Nmstate configuration sections"
            
        return True, "YAML is valid and follows Nmstate schema"
    except yaml.YAMLError as e:
        return False, f"Invalid YAML syntax: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def generate_network_config(user_input, model, tokenizer, device):
    """Generate Nmstate YAML configuration from natural language input"""
    try:
        # Create a pipeline for text generation
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # Prepare prompt
        prompt = f"Convert this network command to Nmstate YAML:\n{user_input}\n---\n"
        
        # Generate YAML
        response = generator(
            prompt,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Clean up generated text
        yaml_str = response[0]["generated_text"].replace(prompt, "").strip()
        
        # Validate YAML
        is_valid, message = validate_yaml(yaml_str)
        if not is_valid:
            return yaml_str, "⚠️ " + message, "error"
            
        return yaml_str, "✅ Valid Nmstate configuration", "success"
        
    except Exception as e:
        return str(e), "❌ Generation failed", "error"


def save_feedback(command, yaml_config, rating, feedback_file):
    """Save user feedback for model improvement"""
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode="a", newline="\n", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["command", "yaml_config", "rating", "timestamp"])
        writer.writerow([command, yaml_config, rating, gr.utils.get_timestamp()])
    return "Thank you for your feedback! This helps improve the model."

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def main():
    parser = argparse.ArgumentParser(description="Network Configuration Model UI")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--feedback_path",
        type=str,
        default="feedback.csv",
        help="Path to save user feedback"
    )
    args = parser.parse_args()

    # Load model
    tokenizer, model = load_model(args.model_path)
    device = 0 if torch.cuda.is_available() else -1

    # Create Gradio interface
    with gr.Blocks(title="Network Configuration Assistant") as interface:
        gr.Markdown("# Network Configuration Assistant")
        gr.Markdown("Convert natural language commands into Nmstate configurations")
        
        with gr.Row():
            with gr.Column():
                command = gr.Textbox(
                    label="Network Command",
                    placeholder="Example: Configure eth0 with IP 192.168.1.10/24 and gateway 192.168.1.1"
                )
                examples = gr.Examples(
                    examples=[
                        ["Create a Linux bridge br0 using eth1 and eth2"],
                        ["Set up VLAN 100 on eth0"],
                        ["Configure eth1 with static IP 10.0.0.5/24"],
                        ["Add DNS server 8.8.8.8"]
                    ],
                    inputs=command
                )
                generate_btn = gr.Button("Generate Configuration")
            
            with gr.Column():
                yaml_output = gr.Code(
                    label="Generated Nmstate YAML",
                    language="yaml"
                )
                status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
        with gr.Row():
            rating = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=5,
                label="Rate the generated configuration (1-5)"
            )
            submit_btn = gr.Button("Submit Feedback")
            feedback = gr.Textbox(label="Feedback Status", interactive=False)

        def on_submit(cmd, yaml_cfg, rating):
            return save_feedback(cmd, yaml_cfg, rating, args.feedback_path)

        generate_btn.click(
            fn=generate_network_config,
            inputs=[command, model, tokenizer, device],
            outputs=[yaml_output, status]
        )
        
        submit_btn.click(
            fn=on_submit,
            inputs=[command, yaml_output, rating],
            outputs=[feedback]
        )



if __name__ == "__main__":
    main()
