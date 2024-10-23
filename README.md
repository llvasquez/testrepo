1. Create virtual environment
python3 -m venv pytorch_env

Activate the virtual environment
source pytorch_env/bin/activate

Install PyTorch using Pip 
pip3 install torch torchvision torchaudio

Downgrade NumPy to a 1.x Version

Install Watchdog module using Pip
pip3 install watchdog

First clone the repository
git clone https://github.com/VikParuchuri/surya.git
cd surya


Install dependencies
pip install torch torchvision

Install the package in editable mode:
pip install -e .


run python to verify the installation
python3
import surya

If you get no error it installed properly


Make sure the .py file you want to use is in the surya folder


to run app using streamlit
streamlit run .py file
