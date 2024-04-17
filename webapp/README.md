# Nmstate UI Demo
![Screenshot from 2024-04-17 18-20-01](https://github.com/jona42-ui/nlpdemo/assets/78595738/03847966-0875-4b4e-875c-a67ffbc1cb52)



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

flask --app app.y run
```

2. Open your web browser and go to [http://localhost:5000](http://localhost:5000).

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
├── requirements.txt
├── static
└── templates
    ├── image.html
    └── index.html
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- Developed by [thembo jonathan](https://github.com/jona42-ui)
- Nmstate library: [Nmstate ](https://github.com/nmstate/nmstate)
- NetVisor tool: [NetVisor](https://github.com/ffmancera/NetVisor)
