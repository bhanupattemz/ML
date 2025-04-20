def find_s_algorithm_without_libraries(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
    data = [line.strip().split(',') for line in lines]
    features = [line[:-1] for line in data]
    target = [line[-1] for line in data]

    hypothesis = ["Ø"] * len(features[0])
    
    for i, example in enumerate(features):
        if target[i] == "Yes": 
            for j in range(len(hypothesis)):
                if hypothesis[j] == "Ø":
                    hypothesis[j] = example[j]
                elif hypothesis[j] != example[j]:
                    hypothesis[j] = "?" 
    return hypothesis
hypothesis = find_s_algorithm_without_libraries(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Experiment1\fruit.csv")
print("Final Hypothesis:", hypothesis)