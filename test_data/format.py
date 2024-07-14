import json
x = json.load(open('test_data.json', 'r'))
ls = []

for p in x:
    ls.append({'query' : p['question'], 'response' : " "})

json.dump(ls, open('test_data_list.json', 'w'))