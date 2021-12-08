filename='DeepLabv3_ResNet101_20.ckpt'

if [ ! -f $filename ]; then
	wget https://www.dropbox.com/s/hq2jovpb0vk6v99/DeepLabv3_ResNet101_25.ckpt?dl=1 -O $filename
fi

python3 ./p2/inference.py --img_dir $1 --save_dir $2 --ckpt $filename
