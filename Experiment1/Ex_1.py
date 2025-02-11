import pandas as pd


def find_s_algorithm(filename):
    data = pd.read_csv(filename)
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values
    hypothesis = ["Ø"] * features.shape[1]
    for i, example in enumerate(features):
        if target[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] == "Ø":
                    hypothesis[j] = example[j]
                elif hypothesis[j] != example[j]:
                    hypothesis[j] = "?"
    return hypothesis
print(find_s_algorithm("fruit.csv"))