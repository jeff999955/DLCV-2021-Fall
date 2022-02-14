if [ $# -lt 1 ]; then
    echo usage: $0 \$image_path
    exit 0
fi

cur_path=`realpath .` # the root of our repository

mkdir -p yolov5_inf/inference/images
python3 ./yolov5_inf/inference/move_npy.py \
    --images_path $1 \
    --target_path yolov5_inf/inference/images/
# $1 is the test dir of image, ex: skull/test

cd ./yolov5_inf/inference
gdown --id 1jGGDfVRe6k9ALgUvE24knVD3MS72Jd1- 

python3 ./../detect.py \
    --source images/ \
    --weights detect.pt --save-txt --name detect 

python3 ./../detect_classify.py \
    --source images/ \
    --weights classify.pt \
    --save-txt --name classify --img 512 

python3 ./case_classify.py \
    --classify_labels_path ../runs/detect/classify/labels \
    --out_path pred.csv \
    --detect_labels_path ../runs/detect/detect/labels \
    --img_path images/

python3 ./post_processing.py \
    --input_file pred.csv \
    --output_file post.csv

cp post.csv $cur_path
