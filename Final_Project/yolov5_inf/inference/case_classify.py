import os
import csv
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--classify_labels_path', type=str, default='./runs/detect/exp17/labels', help='')
parser.add_argument('--out_path', type=str, default='pred_exp25_exp17case.csv', help='')
parser.add_argument('--detect_labels_path', type=str, default='./runs/detect/exp25/labels', help='')
parser.add_argument('--img_path', type=str, default='../skull/test_all_images', help='')
config = parser.parse_args()


# classify_labels_path = './runs/detect/exp17/labels'
# out_path = 'pred_exp25_exp17case.csv'
# detect_labels_path = './runs/detect/exp25/labels'
# img_path = '../skull/test_all_images'

classify_labels_path  = config.classify_labels_path
out_path              = config.out_path
detect_labels_path    = config.detect_labels_path
img_path              = config.img_path

cases_dict = {}
for file in os.listdir(classify_labels_path):
    case_id = file[:20]
    cases_dict[str(case_id)] = 0

for file in os.listdir(classify_labels_path):
    case_id = file[:20]
    f = open(os.path.join(classify_labels_path, file), 'r')
    lines = f.readlines()
    for line in lines:
        if line.split(' ')[0] == '1':  # without
            cases_dict[str(case_id)] += 1
        elif line.split(' ')[0] == '2':  # with
            cases_dict[str(case_id)] -= 1

cases = []
for (case_id, count) in cases_dict.items():
    print(case_id, ": ", count)
    if not case_id in cases and count <= 0:
        cases.append(case_id)
print(len(cases))

datas = sorted(os.listdir(img_path))
print(len(datas))
with open(out_path, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['id', 'label', 'coords'])

    for data in datas:
        case_id = data[:20]
        coords = ''
        label_path = os.path.join(detect_labels_path, data.split('.')[0] + '.txt')
        if os.path.isfile(label_path):
            if case_id in cases:
                f = open(label_path, 'r')
                lines = f.readlines()
                for line in lines:
                    if not coords == '':
                        coords += ' '
                    coords += str(round(float(line.split(' ')[1]) * 512))
                    coords += ' '
                    coords += str(round(float(line.split(' ')[2]) * 512))
                label = 1
                writer.writerow([data.split('.')[0], label, coords])
            else:
                writer.writerow([data.split('.')[0], '0', ''])
        else:
            if case_id in cases:
                label = -1
            else:
                label = 0
            writer.writerow([data.split('.')[0], label, coords])
