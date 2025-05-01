
# Installation Guide for Deep Researcher


This document provides step-by-step instructions for installing and setting up the Deep Researcher application. Follow the instructions carefully to ensure a successful installation.


## Prerequisites


Before you begin, ensure you have the following installed on your system:


- **Python 3.7 or higher**: You can download it from [python.org](https://www.python.org/downloads/).
- **Git**: If you don't have Git installed, you can download it from [git-scm.com](https://git-scm.com/downloads).
- **Curl**: This is typically pre-installed on macOS and Linux. For Windows, you can use [curl for Windows](https://curl.se/windows/).


## Installation Steps


### 1. Clone the Repository


Open your terminal and run the following commands to clone the Deep Researcher repository:


```bash
git clone https://github.com/Sallyliubj/Deep-Researcher.git
cd Deep-Researcher
```


### 2. Create and Activate a Virtual Environment


It is recommended to use a virtual environment to manage dependencies. Run the following commands:


```bash
# Create a virtual environment
python -m venv .venv


# Activate the virtual environment
# For macOS/Linux
source .venv/bin/activate


# For Windows
.\.venv\Scripts\activate
```


### 3. Install Required Dependencies


Once the virtual environment is activated, install the required Python packages using pip:


```bash
pip install -r requirements.txt
```


### 4. Set Up Ollama for Local LLM


Deep Researcher uses Ollama for local AI model inference. Follow these steps to set it up:


1. **Install Ollama** (if not already installed):


  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```


2. **Pull the Gemma model**:


  ```bash
  ollama pull gemma3:1b
  ```


### 5. Configure Environment Variables


Create a `.env` file in the root directory of the project to store your API keys and other configuration settings. You can use the provided `.env.example` as a template:


```bash
cp .env.example .env
```


Edit the `.env` file and add your API keys and any other necessary configurations.


### 6. Start the Application


To start the Deep Researcher application, run the following command:


```bash
export PYTHONPATH=$(pwd)
streamlit run app/main.py
```


Alternatively, you can use the provided shell script:


```bash
./run.sh
```


### 7. Access the Application


Open your web browser and navigate to `http://localhost:8501` to access the Deep Researcher application.


## Troubleshooting


If you encounter any issues during installation or while running the application, consider the following:


- Ensure that all dependencies are correctly installed.
- Check that your virtual environment is activated.
- Verify that your API keys in the `.env` file are correct and have the necessary permissions.
- Consult the documentation for any specific errors you may encounter.


## Conclusion


You have successfully installed the Deep Researcher application. You can now start conducting comprehensive research using the various features provided by the application. For further assistance, refer to the [README.md](README.md) or the project's documentation.
```


This `INSTALLATION.md` file provides clear and detailed instructions for setting up the Deep Researcher application, ensuring that users can follow along easily.
