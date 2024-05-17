import pandas as pd
from transformers import pipeline

data = pd.read_csv('data/fwdnlpdata/reported_loss_scrapings_long.csv')

classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')


data['report_type'] = data['report_type'].map({2:-1,0:1,1:1,3:1,4:1,5:1,6:1,7:1,})
data['report_type'] = data['report_type'].map({2:0, 1:1})
labels = ['haiti']
# labels = ['haiti']
tp = 0
tn = 0
fp = 0
fn = 0
correct = 0
for i, row in data.iterrows():
    if i % 100 == 0:
        print(i)
    results = classifier(row["report"], labels)
    if results["scores"][0] > 0.5:
        label = 0
    else:
        label = 1
    
    if label == 0 and row['report_type'] == 0:
        tp += 1
        correct += 1
    elif label == 1 and row['report_type'] == 1:
        tn += 1
        correct += 1
    elif label == 0 and row['report_type'] == 1:
        fp += 1
    else:
        fn += 1

print("%d / %d" % (correct, data.shape[0]))
print("True Positive:", tp)
print("True Negative:", tn)
print("False Positive:", fp)
print("False Negative:", fp)
print("FP Rate:", fp / (tn + fp))
