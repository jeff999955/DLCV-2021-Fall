if [ ! -f ckpt.tar.gz ]; then
	wget https://www.dropbox.com/s/58pfcdz7x2f8c0t/ckpt.tar.gz?dl=1 -O ckpt.tar.gz
	tar zxvf ckpt.tar.gz
fi

python3 ./p1/inference.py --test_dir $1 --out_csv $2 --ckpt_dir `realpath .`
