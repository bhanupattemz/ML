import csv
import math

def calculate_entropy(data):
    value_counts = {}
    for val in data:
        if val not in value_counts:
            value_counts[val] = 0
        value_counts[val] += 1
    
    entropy = 0
    for count in value_counts.values():
        prob = count / len(data)
        entropy -= prob * math.log2(prob)
    return entropy


def split_data(data, attribute):
    splits = {}
    for row in data:
        key = row[attribute]
        if key not in splits:
            splits[key] = []
        splits[key].append(row)
    return splits

def id3(data, attributes, target_col):
    target_values = [row[target_col] for row in data]
    if len(set(target_values)) == 1:
        return target_values[0]
    if not attributes:
        return max(set(target_values), key=target_values.count)
    base_entropy = calculate_entropy(target_values)
    best_gain = -1
    best_attr = None
    best_splits = None
    for attr in attributes:
        splits = split_data(data, attr)
        weighted_entropy = 0
        for subset in splits.values():
            target_subset = [row[target_col] for row in subset]
            weighted_entropy += (len(target_subset) / len(data)) * calculate_entropy(target_subset)
        
        info_gain = base_entropy - weighted_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_attr = attr
            best_splits = splits
    
    tree = {headers[best_attr]: {}}
    for key, subset in best_splits.items():
        subtree = id3(subset, [attr for attr in attributes if attr != best_attr], target_col)
        tree[headers[best_attr]][key] = subtree
    
    return tree

filename = r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment3\games.csv"
data = []

with open(filename, newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)  
    for row in reader:
        if len(row) != len(headers):
            continue
        data.append(row)

attributes = list(range(len(headers) - 1))  
target_col = len(headers) - 1  

decision_tree = id3(data, attributes, target_col)

print("Decision Tree:")
for key, value in decision_tree.items():
    print(f"{key} => {value}")
