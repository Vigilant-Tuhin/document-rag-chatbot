#!/bin/sh

# Start Ollama in background
ollama serve &

# Wait for the server to actually be ready, instead of a fixed sleep that
# could be too short on a slow machine or wastefully long on a fast one.
echo "Waiting for Ollama server to start..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done
echo "Ollama server is up."

# Pull the model
echo "Pulling llama3.1:8b model..."
ollama pull llama3.1:8b

# Keep container running (waits on the backgrounded `ollama serve` process)
wait
