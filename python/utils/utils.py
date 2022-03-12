import csv
import numpy as np

def load_csv(path, skip_header=True):
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = None
        if skip_header: headers = next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)
        
    return rows, headers


def one_hot(y, cnt):
    return np.eye(cnt)[np.array(y).astype(int)]

