#!/bin/bash

echo "🔧 Setting up Cautious RAG development environment..."

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Create necessary directories
mkdir -p data/raw data/processed results/figures results/metrics

echo "✅ Setup complete!"
echo "👉 Activate environment with: source venv/bin/activate"