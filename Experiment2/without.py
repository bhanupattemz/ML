def candidate_elimination_without_libraries(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = [line.strip().split(',') for line in lines]
    features = [line[:-1] for line in data]
    target = [line[-1] for line in data]

    general_hypothesis = ['?' for _ in range(len(features[0]))]
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


general, specific = candidate_elimination_without_libraries(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment2\zoo.csv")
print("Most General Hypothesis:", general)
print("Most Specific Hypothesis:", specific)