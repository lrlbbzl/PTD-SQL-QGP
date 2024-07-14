
import json
gt = json.load(open('../train_data/ft_valid_data.json', 'r'))
pred = json.load(open('./predictions_gpt.json', 'r'))


qs = [q['query'] for q in gt]
gt_ls = [q['response'] for q in gt]
pred_ls = [q['predict'] for q in pred]

from sklearn.metrics import accuracy_score, classification_report

print(classification_report(gt_ls, pred_ls))
print(accuracy_score(gt_ls, pred_ls))