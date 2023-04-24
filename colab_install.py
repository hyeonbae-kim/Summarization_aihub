import os

PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)

os.chdir(PROJECT_DIR)
os.system("pip install -r requirements.txt")
os.system("pip install Cython")
os.system("python src/others/install_mecab.py")
os.system("pip install -r requirements_prepro.txt")
os.system("pip install transformers")
os.system("pip install pyrouge==0.1.3")