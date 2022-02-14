import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='./pred_exp25_exp17case.csv', help='')
parser.add_argument('--output_file', type=str, default='post_exp25_exp17case.csv', help='')
config = parser.parse_args()

input_file = config.input_file
output_file = config.output_file

df = pd.read_csv(input_file)
cases = list(df['id'])
ids = {}
counts = {}
i = 0
for index, row in df.iterrows():
    i += 1
    # id: str, label: int, coords: float
    case, label, coords = row
    cnt = 0
    case = case.split('_')
    id = '_'.join(case[:-1])
    if id not in ids:
        ids[id] = [case[-1]]
    else:
        ids[id].append(case[-1])
    if isinstance(coords, str):
        cnt = len(coords.split())
    if id not in counts:
        counts[id] = [cnt]
    else:
        counts[id].append(cnt)


def cont(l):
    cur, n = -1, len(l)
    for i in range(n):
        if l[i]:
            if cur == -1: cur = i
            else: return True
        else:
            cur = -1
    return False


with open(output_file, 'w', newline='') as f:
    print('id,label,coords', file = f)
    for id in counts:
        cnt = counts[id]
        result = cont(cnt)
        print(id, cnt, result)
        if result:
            print(df[df['id'].str.startswith(id)].to_csv(index = False, header = False).rstrip(), file = f)
        else:
            for case in ids[id]:
                print(f"{id}_{case},0,", file = f)
