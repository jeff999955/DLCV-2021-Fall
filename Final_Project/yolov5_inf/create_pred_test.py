import os
import csv
import json
import glob

out_path = 'pred_exp23.csv'
# records_path = '../skull/records_train.json'
labels_path = './runs/detect/exp23/labels'
img_path = '../skull/all_images'

cases = []
for file in os.listdir(labels_path):
    case_id = file[:20]
    if not case_id in cases:
        cases.append(case_id)
print(len(cases))
# for id, data in datas['datainfo'].items():
#     case_id = id[:20]
#     if glob.glob(os.path.join(labels_path, case_id))
#     cases[case_id] = os.path.join(labels_path, case_id)

datas = os.listdir(img_path)
print(len(datas))
with open(out_path, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['id', 'label', 'coords'])

    case = ''
    for data in datas:
        case_id = data[:20]
        coords = ''
        label_path = os.path.join(labels_path, data.split('.')[0] + '.txt')
        # print(label_path)
        # input()
        if os.path.isfile(label_path):
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
            # print(glob.glob(os.path.join(labels_path, case_id)))
            # input()
            if case_id in cases:
                label = -1
            else:
                label = 0
            writer.writerow([data.split('.')[0], label, coords])

        # for coord in data['coords']:
        #     if not coords == '':
        #         coords += ' '
        #     coords = coords + str(coord[0]) + ' ' + str(coord[1])
        # writer.writerow([id, data['label'], coords])
