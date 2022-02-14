import os
import csv
import json

out_path = 'gt.csv'
records_path = '../skull/records_train.json'

with open(os.path.join(records_path), 'r') as jsonfile:
    datas = json.load(jsonfile)

with open(out_path, 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['id', 'label', 'coords'])

    for id, data in datas['datainfo'].items():
        coords = ''
        for coord in data['coords']:
            if not coords == '':
                coords += ' '
            coords = coords + str(coord[0]) + ' ' + str(coord[1])
        writer.writerow([id, data['label'], coords])