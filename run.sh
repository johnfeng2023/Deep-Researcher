#!/bin/bash
export PYTHONPATH=$(pwd)
# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama is not running. Starting Ollama..."
    # Try to start Ollama (this command might vary depending on how Ollama was installed)
    ollama serve > /dev/null 2>&1 &
    
    # Wait for Ollama to start
    echo "Waiting for Ollama to start..."
    sleep 5
    
    # Check again if Ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo "Failed to start Ollama. Please start it manually before running the application."
        exit 1
    fi
    
    echo "Ollama started successfully."
fi

# Check if the Gemma model is available
if ! curl -s http://localhost:11434/api/tags | grep -q "gemma3:1b"; then
    echo "Gemma 3 1B model not found. Pulling the model..."
    ollama pull gemma3:1b
    
    if [ $? -ne 0 ]; then
        echo "Failed to pull Gemma 3 1B model. Please pull it manually with 'ollama pull gemma3:1b'."
        exit 1
    fi
    
    echo "Gemma 3 1B model pulled successfully."
fi

# Run the Streamlit app
echo "Starting Deep Researcher..."
streamlit run app/main.py 