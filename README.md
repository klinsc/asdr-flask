# asdr-flask

**How to create conda environment for asdr-flask**

**If the disk space is low, use `conda clean -a`**

## 1) First, we create a conda environment with python 3.10.12

`conda create -n asdr python=3.10.12`

## 2) Then, we activate the environment

`conda activate asdr`

## 3) Next, we install all CUDA related packages by official pytorch https://pytorch.org/get-started/previous-versions/

`conda install pytorch==1.12.1 -c pytorch`

## 4) Then, we install all other packages from YOLOv5 official requirements.txt

`conda install poppler`

`pip install pandas flask flask_cors opencv-python pdf2image prisma requests`

`pip install flask[async]`

`pip install ultralytics --no-cache-dir`

## 5) Generate prima client

`prisma generate`

## 6) Run the server

`python asdr.py`
