1. Create virtual environment
python3 -m venv pytorch_env

Activate the virtual environment
source pytorch_env/bin/activate

Install PyTorch using Pip 
pip3 install torch torchvision torchaudio

Downgrade NumPy to a 1.x Version
pip3 install "numpy<2"

python3 -m venv my_env (create virtual environment)
source my_env/bin/activate (activate virtual environment)

pip3 install Watchdog 
pip3 install streamlit

Downgrade NumPy to a 1.x Version
pip3 install "numpy<2"

Install PyTorch using Pip 
pip3 install torch torchvision torchaudio

clone the repository
git clone https://github.com/VikParuchuri/surya.git
cd surya

pip3 install surya-ocr

Install the package in editable mode:
pip install -e .


run python to verify the installation
python3
import surya

If you get no error it installed properly

Make sure the .py file you want to use is in the surya folder

exit python - exit()

to run app use
streamlit run surya_ocr.py

to deactivate virtual environment
deactivate




Make sure the .py file you want to use is in the surya folder


to run app using streamlit
streamlit run .py file
