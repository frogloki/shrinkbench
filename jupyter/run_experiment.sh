
VENV_NAME=".venv"
MAIN_SCRIPT="shrinkbench/jupyter/experiment.py"

echo "--- Starting Project Setup ---"

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment in '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Exiting."
        exit 1
    fi
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

echo "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Exiting."
    exit 1
fi

echo "--- Setup Complete ---"
echo ""

echo ">>> Running main script: $MAIN_SCRIPT <<<"
echo ""

python $MAIN_SCRIPT
echo ""
echo "--- Script Finished. Deactivating environment. ---"
deactivate