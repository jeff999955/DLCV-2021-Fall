import os
import csv
import json
import glob

# out_path = 'pred_exp17.csv'
# records_path = '../skull/records_train.json'
labels_path = './yolov5/runs/detect/exp17/labels'
# img_path = '../skull/test_all_images'

cases_dict = {}
for file in os.listdir(labels_path):
    case_id = file[:20]
    cases_dict[str(case_id)] = 0

for file in os.listdir(labels_path):
    case_id = file[:20]
    f = open(os.path.join(labels_path, file), 'r')
    lines = f.readlines()
    for line in lines:
        if line.split(' ')[0] == '1':  # without
            cases_dict[str(case_id)] += 1
        elif line.split(' ')[0] == '2':  # with
            cases_dict[str(case_id)] -= 1

# print(cases_dict)
cases = []
for (case_id, count) in cases_dict.items():
    print(case_id, ": ", count)
    if not case_id in cases and count <= 0:
        cases.append(case_id)
print(len(cases))
# for id, data in datas['datainfo'].items():
#     case_id = id[:20]
#     if glob.glob(os.path.join(labels_path, case_id))
#     cases[case_id] = os.path.join(labels_path, case_id)

# datas = os.listdir(img_path)
# print(len(datas))
# with open(out_path, 'w', newline='') as out_file:
#     writer = csv.writer(out_file)
#     writer.writerow(['id', 'label', 'coords'])

#     case = ''
#     for data in datas:
#         case_id = data[:20]
#         coords = ''
#         label_path = os.path.join(labels_path, data.split('.')[0] + '.txt')
#         # print(label_path)
#         # input()
#         fracture_count = 0
#         if os.path.isfile(label_path):
#             if case_id in cases:
#                 writer.writerow([data.split('.')[0], '1', ''])
#             else:
#                 writer.writerow([data.split('.')[0], '0', ''])
#         else:
#             # print(glob.glob(os.path.join(labels_path, case_id)))
#             # input()
#             if case_id in cases:
#                 label = 1
#             else:
#                 label = 0
#             writer.writerow([data.split('.')[0], label, ''])

#         # for coord in data['coords']:
#         #     if not coords == '':
#         #         coords += ' '
#         #     coords = coords + str(coord[0]) + ' ' + str(coord[1])
#         # writer.writerow([id, data['label'], coords])

out_path = './yolov5/pred_exp14_exp17case_rrrrr.csv'
# records_path = '../skull/records_train.json'
labels_path = './yolov5//runs/detect/exp14/labels'
img_path = './skull/test_all_images'


datas = os.listdir(img_path)
print(len(datas))
with open(out_path, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['id', 'label', 'coords'])

    for data in datas:
        case_id = data[:20]
        coords = ''
        label_path = os.path.join(labels_path, data.split('.')[0] + '.txt')
        # print(label_path)
        # input()
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
