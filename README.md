# asdr-flask

**How to create conda environment for asdr-flask**

**If the disk space is low, use `conda clean -a`**

## 1) First, we create a conda environment with python 3.10.12

`conda create -n asdr python=3.11.4`

## 2) Then, we activate the environment

`conda activate asdr`

## 3) Next, we install all CUDA related packages by official pytorch https://pytorch.org/get-started/previous-versions/

`conda install pytorch==1.12.1 -c pytorch`

## 4) Then, we install all other packages from YOLOv5 official requirements.txt

`conda install poppler uwsgi`

`pip install pandas flask flask_cors opencv-python pdf2image prisma requests`

`pip install flask[async]`

`pip install ultralytics --no-cache-dir`

## 5) Generate prima client

`prisma generate`

## 6) Run the server

`python asdr.py`

# Concate all commands

`docker run -dit --name ubuntu-2 --publish 80:80 ubuntu:22.04`

`conda create -n asdr python=3.11.4 && conda activate asdr && conda install -y pytorch==1.12.1 -c pytorch && conda install -y poppler uwsgi && pip install pandas flask flask_cors opencv-python pdf2image prisma requests && pip install flask[async] && pip install ultralytics && prisma generate && python asdr.py`

`apt-get update && apt-get install -y wget`

`mkdir -p ~/miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && rm -rf ~/miniconda3/miniconda.sh && ~/miniconda3/bin/conda init bash`

`apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools`

`nano /etc/systemd/system/asdr.service`
`[Unit]
Description=uWSGI instance to serve asdr-flask
After=network.target

[Service]
User=root
Group=www-data
WorkingDirectory=/home/asdr-flask
Environment="PATH=/root/miniconda3/envs/asdr/bin"
ExecStart=/root/miniconda3/envs/asdr/bin/uwsgi --ini asdr.ini

[Install]
WantedBy=multi-user.target`

`apt-get update && apt-get install ffmpeg libsm6 libxext6 -y`

`conda create -n asdr python=3.11.4 && conda activate asdr`
`conda install -y poppler uwsgi`
`pip install -r requirements.txt`
`pip install flask[async]`
`conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`
`nano ~/.bashrc`
add `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`
`source ~/.bashrc`

Permission denied while connecting to upstream nginx flask
`https://stackoverflow.com/questions/70111791/nginx-13-permission-denied-while-connecting-to-upstream`

Install poppler
`https://askubuntu.com/questions/1240998/how-to-install-poppler-0-73-on-ubuntu-20-04-any-change-since-18-04`
