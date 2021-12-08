gdown --id 13FunFbdNSEbi_mGigHiG8Rf9SEuB3f_K -O ckpt.zip
unzip ckpt.zip
python3 p1/inference.py --test_dir $1 --out_csv $2
