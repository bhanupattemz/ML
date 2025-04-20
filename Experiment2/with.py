import pandas as pd
def candidate_elimination_with_libraries(filename):
    data = pd.read_csv(filename)
    features = data.iloc[:, :-1].values
    target = data.iloc[:, -1].values

    general_hypothesis = ['?' for _ in range(features.shape[1])]
    specific_hypothesis = features[0].copy()

    for i, example in enumerate(features):
        if target[i] == 'Yes':
            for j in range(len(general_hypothesis)):
                if general_hypothesis[j] == '?':
                    general_hypothesis[j] = example[j]
                elif general_hypothesis[j] != example[j]:
                    general_hypothesis[j] = '?'
        else:
            for j in range(len(specific_hypothesis)):
                if specific_hypothesis[j] != example[j]:
                    specific_hypothesis[j] = '?'

    return general_hypothesis, specific_hypothesis

general, specific = candidate_elimination_with_libraries(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment2\zoo.csv")
print("Most General Hypothesis:", general)
print("Most Specific Hypothesis:", specific)