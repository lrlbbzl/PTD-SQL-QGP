import json

x = json.load(open('results.json', 'r'))

ls = []
for p in x:
    temp = p['predict']
    temp = temp[temp.find('Type: ') + len('Type: ') : ]
    if temp.startswith('Set operation'):
        pred = 'Set operation'
    elif temp.startswith('Filter problem'):
        pred = 'Filter problem'
    elif temp.startswith('Combination operation'):
        pred = 'Combination operation'
    elif temp.startswith('Other'):
        pred = 'Other simple problem'
    else:
        print('wtf')
    ls.append({'query' : p['query'], 'predict' : pred})

json.dump(ls, open('predictions.json', 'w'))