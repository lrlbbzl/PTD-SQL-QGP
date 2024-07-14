import json
import os
x = json.load(open('./gpt_test_type.json', 'r'))

mp = {'combination' : [], 'filter' : [], 'complex' : [], 'simple' : []}
y = json.load(open('../test_data.json', 'r'))

for yy in y:
    v = x[yy['question']]
    if v == 'Filtering problems':
        mp['filter'].append(yy)
    elif v == 'Combination operations':
        mp['combination'].append(yy)
    elif v == 'Other simple problems':
        mp['simple'].append(yy)
    else:
        mp['complex'].append(yy)

for k, v in mp.items():
    dir = './{}'.format(k)
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, '{}.json'.format(k))
    json.dump(v, open(path, 'w'))
