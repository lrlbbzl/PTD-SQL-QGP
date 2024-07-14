import json
import random

data = json.load(open('train_data.json', 'r'))
x = json.load(open('ft_data.json', 'r'))

mp = {}
for dic in x:
    if dic['response'] not in mp:
        mp.update({dic['response'] : []})
    mp[dic['response']].append(dic['query'])

# 1
complex_valid_data = []
complex_data = []
for p in data:
    if 'union' in p['gold'].lower() or 'intersect' in p['gold'].lower() or 'except' in p['gold'].lower():
        complex_data.append(p['question'])
        if p['question'] not in mp['Set operation']:
            complex_valid_data.append(p['question'])
print('Set operation: {}'.format(len(complex_valid_data)))

# 2
combination_valid_data = []
combination_data = []
for p in data:
    if 'group by' in p['gold'].lower() or 'GROUP BY' in p['gold'].lower():
        combination_data.append(p['question'])
        if p['question'] not in mp['Combination operation']:
            combination_valid_data.append(p['question'])
print('Combination operation: {}'.format(len(combination_valid_data)))

#3
filter_valid_data = []
filter_data = []
for p in data:
    if 'where' in p['gold'].lower() and p['question'] not in complex_data and p['question'] not in combination_data:
        filter_data.append(p['question'])
        if p['question'] not in mp['Filter problem']:
            filter_valid_data.append(p['question'])
print('Filter problem: {}'.format(len(filter_valid_data)))


#4
simple_valid_data = []
for p in data:
    if p['question'] not in complex_data and p['question'] not in combination_data and p['question'] not in filter_data:
        if p['question'] not in mp['Other simple problem']:
            simple_valid_data.append(p['question'])
print('Other simple problem: {}'.format(len(simple_valid_data)))

## sample

complex_select_valid = random.sample(complex_valid_data, 40)
combination_select_valid = random.sample(combination_valid_data, 80)
filter_select_valid = random.sample(filter_valid_data, 100)
simple_select_valid = random.sample(simple_valid_data, 80)
d = []
for a in complex_select_valid:
    d.append({
        'query' : a,
        'response' : 'Set operation'
    })    
for a in combination_select_valid:
    d.append({
        'query' : a,
        'response' : 'Combination operation'
    })
for a in filter_select_valid:
    d.append({
        'query' : a,
        'response' : 'Filter problem'
    })
for a in simple_select_valid:
    d.append({
        'query' : a,
        'response' : 'Other simple problem'
    })
json.dump(d, open('ft_valid_data.json', 'w'))