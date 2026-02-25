#!/bin/sh

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate the virtual environment
. .venv/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo "Minifying static assets..."
python -m rjsmin < script.js > static/script.min.js
python -m rcssmin < styles.css > static/styles.min.css

echo "Starting Flask debug server"
python -u -m flask --app main run --host=0.0.0.0 -p ${PORT:-8080} --debug
