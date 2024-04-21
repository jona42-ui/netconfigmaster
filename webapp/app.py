from flask import Flask, render_template, request, send_file
import libnmstate
from libnmstate.schema import Interface
import subprocess
import json
import yaml

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    
    # Query network state
    #net_state = libnmstate.show()
    #nmstate_output_text = ""
    #for iface_state in net_state[Interface.KEY]:
    #    nmstate_output_text += json.dumps(iface_state) + "\n"

    net_state = libnmstate.show()
    nmstate_output_text = yaml.dump(net_state)


    # network state image
    #netvisor_output = subprocess.run(['/home/thembo/fedora/dev/NetVisor/target/debug/netvisor', '--input', json.dumps(net_state)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #netvisor_output_text = netvisor_output.stdout.decode('utf-8')

    # Get user input
    user_input = request.form['inputText']


    # Render template 
    return render_template('index.html', input_text=user_input, nmstate_output=nmstate_output_text,)
@app.route('/show_image', methods=['GET'])
def generate_image():
    try:
        # Generate the image file
        output_path = '/static/output.png'
        netvisor_output = subprocess.run(['/home/thembo/fedora/dev/NetVisor/target/debug/netvisor', 'show', '--file', output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if netvisor_output.returncode == 0:
            # Send the generated image file as a response
            return send_file(output_path, mimetype='image/png')
        else:
            return "Error generating image"
    except subprocess.CalledProcessError:
        return "Error generating image"

# Route for displaying the image template
@app.route('/show_image', methods=['GET'])
def show_image():
    return render_template('image.html')


if __name__ == '__main__':
    app.run(debug=True)
