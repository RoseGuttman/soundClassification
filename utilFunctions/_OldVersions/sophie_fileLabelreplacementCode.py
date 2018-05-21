import csv

data = {}
fields = ['# YTID', ' start_seconds', ' end_seconds', ' positive_labels', 'index']

with open('balanced_train_segments.csv', 'r') as f:
    for line in csv.DictReader(f):
        if ' positive_labels' in line.keys():
            data[line[' positive_labels']] = line

with open('class_labels_indices.csv', 'r') as f:
    for line in csv.DictReader(f):
        if ' "' + line['mid'] not in data.keys():
            data[' "'+line['mid']] = {}
        data[' "' + line['mid']]['index'] = line['index']

with open('output.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames = fields, extrasaction='ignore')
    for line in data.values():
        writer.writerow(line)