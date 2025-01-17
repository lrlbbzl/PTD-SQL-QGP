import json
import random

# x = json.load(open('train_data.json', 'r'))

# complex_data = []
# complex_select_data = []

# for p in x:
#     if 'union' in p['gold'].lower() or 'intersect' in p['gold'].lower() or 'except' in p['gold'].lower():
#         complex_data.append(p['question'])
# complex_select_data = random.sample(complex_data, 400)

# combination_data = []
# combination_select_data = []

# for p in x:
#     if 'group by' in p['gold'].lower() or 'GROUP BY' in p['gold'].lower():
#         combination_data.append(p['question'])
# combination_select_data = random.sample(combination_data, 400)

# filter_data = []
# filter_select_data = []

# for p in x:
#     if 'where' in p['gold'].lower() and p['question'] not in complex_data and p['question'] not in combination_data:
#         filter_data.append(p['question'])
# filter_select_data = random.sample(filter_data, 400)


# simple_data = []
# for p in x:
#     if p['question'] not in complex_data and p['question'] not in combination_data and p['question'] not in filter_data:
#         simple_data.append(p['question'])
# simple_select_data = random.sample(simple_data, 400)


# dic = {}
# for k in complex_select_data:
#     dic.update({k : 'Set operation'})
# for k in combination_select_data:
#     dic.update({k : 'Combination operation'})
# for k in filter_select_data:
#     dic.update({k : 'Filter problem'})
# for k in simple_select_data:
#     dic.update({k : 'Other simple problem'})
# json.dump(dic, open('ft_data.json', 'w'))

dic = json.load(open('ft_new_data.json', 'r'))
new_dic = []
for k, v in dic.items():
    new_dic.append({'query' : k, 'response' : v})
json.dump(new_dic, open('train_data.json', 'w'))
print(len(new_dic))