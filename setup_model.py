import os
import pathlib


if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  res = os.system(["git clone --depth 1 https://github.com/tensorflow/models"]) 
else:
    print("Models Already exist so ignoring installation model")