if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Installing..."

    python3 -m venv .venv && \
    source .venv/bin/activate && \
    pip install --upgrade pip && \
    pip cache purge && \
    pip install torch diffusers compel peft && \
    pip install uvicorn fastapi python-multipart && \
    pip freeze > requirements.txt
else
    echo "Virtual environment found. Activating..."
fi 

source .venv/bin/activate

echo "Running..."
python3 src/api.py