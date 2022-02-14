# ***我不確定助教到時候給的dataset 資料夾結構長怎樣，要再確認一下***
# 這些指令我都是在inference資料夾裡執行，其他地方執行可能相對位子要改一下

# 把images移到同一個資料夾裡
python3 move_npy.py --images_path {test image路徑} --target_path ./images/

# 執行完下面的，應該會在/runs/detect/detect裡有一個labels資料夾
python3 ../detect.py --source ./images/ --weights detect.pt --save-txt --name detect

# 執行完下面的，應該會在/runs/detect/classify裡有一個labels資料夾
python3 ../detect_classify.py --source ./images/ --weights classify.pt
 --save-txt --name classify --img 512

# 執行完下面的，應該會在同個資料夾下產生一個
python3 case_classify.py --classify_labels_path ../runs/detect/detect/labels \
	                 --out_path pred.csv \
                         --detect_labels_path ../runs/detect/classify/labels \
                         --labels_path ./images/

# 執行完下面的，應該會出現pred_withpost.csv，就是最後要的
python3 post_processing.py --input_file pred.csv --output_file {最後output csv的path}

===================================================================================
# 範例，有一個skull資料夾跟yolov5_inf並排，skull裡有個資料夾是test
# 寫成shell要再看要放到哪
skull/
    test/
        H1_00000008_00000194/
        H1_00000015_00000965/
        ...
yolov5_inf/
    inference/
        images/
        case_classify.py
        classify.pt
        detect.pt
        move_npy.py
        post_processing.py
        readme.txt

python3 move_npy.py --images_path ../../skull/test/ --target_path ./images/
python3 ../detect.py --source ./images/ --weights detect.pt --save-txt --name detect
python3 ../detect_classify.py --source ../../skull/test_all_images/ --weights classify.pt --save-txt --name classify --img 512
python3 case_classify.py --classify_labels_path ../runs/detect/classify/labels --out_path pred.csv --detect_labels_path ../runs/detect/detect/labels --img_path ./images/
python3 post_processing.py --input_file pred.csv --output_file pred_withpost.csv

# 最後出現pred_withpost.csv後可以用下面這個檢查對不對，如果acc跟F1都是1代表對了(上傳github的不用這個
python3 ../for_students_eval.py --pred_file ../post_exp14_exp19case.csv --gt_file pred_withpost.csv