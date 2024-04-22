# Nmstate UI Demo
yaml api
![image](https://github.com/jona42-ui/nlpdemo/assets/78595738/1cf53e68-8769-4050-b0b1-d45e6e2e678e)

![image](https://github.com/jona42-ui/nlpdemo/assets/78595738/fd8e5cad-3988-4dcf-b1a2-767e7dc1bd92)

This is a simple Flask web application that demonstrates the usage of the Nmstate library to interact with network state in Python.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jona42-ui/nlpdemo.git
cd nlpdemo/webapp
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```bash
python app.py

or

flask --app app.py run
```

2. Open your web browser and go to [http://localhost:5000](http://127.0.0.1:5000).

3. Enter natural language commands related to network configuration in the input field and click "Submit" to see the generated network state.

4. Click on "Show Image" to visualize the network topology image.

## NetVisor Integration
![output](https://github.com/jona42-ui/nlpdemo/assets/78595738/11fc56dd-02e5-44f2-b0b5-82c6a4f26257)


This demo application also integrates with NetVisor for network topology visualization. NetVisor is a host network topology visualization tool that generates images representing the network topology.

To use NetVisor with this application:

1. Ensure you have NetVisor installed and available in your environment. see: https://github.com/ffmancera/NetVisor

2. After submitting a network configuration command in the input field, click "Show Image" to generate and display the network topology image using NetVisor.

3. If you encounter any issues with NetVisor integration, ensure that NetVisor is correctly installed and accessible from your environment.

## Directory Structure

```
.
├── app.py
├── __init__.py
├── output.png
├── __pycache__
│   ├── app.cpython-310.pyc
│   └── __init__.cpython-310.pyc
├── README.md
├── requirements.txt
├── static
└── templates
    └── index.html

```

### DEMO


[Screencast from 04-17-2024 07:18:05 PM.webm](https://github.com/jona42-ui/nlpdemo/assets/78595738/d75133a9-946b-42ae-b562-2cec21150d20)



[Screencast from 04-17-2024 07:19:34 PM.webm](https://github.com/jona42-ui/nlpdemo/assets/78595738/ab77485a-3390-4b0e-a7e4-58ef4975cf67)



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Developed by [thembo jonathan](https://github.com/jona42-ui)
- Nmstate library: [Nmstate ](https://github.com/nmstate/nmstate)
- NetVisor tool: [NetVisor](https://github.com/ffmancera/NetVisor)
