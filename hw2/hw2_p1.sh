if [ $# -lt 1 ]; then
    echo usage: $0 target_directory
    exit 1
fi

target_directory=$1
wget https://www.dropbox.com/s/wrvsc4xyyja1g1q/models.zip?dl=1 -O p1.zip
unzip p1.zip
pip install stylegan2_pytorch
python3 p1_generate.py --save_dir $1 --load_from 140
