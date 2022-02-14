import pandas as pd

df = pd.read_csv('./yolov5/pred_exp14_exp17case_rrrrr.csv')
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


with open('./yolov5/post_exp14_exp17case_rrrrr.csv', 'w', newline='') as f:
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
