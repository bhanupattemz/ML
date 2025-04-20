with open("c:/Users/bhanu/OneDrive/Desktop/@jntua/ML_lab/Exam/fruit.csv","r") as file:
    data=file.readlines()
    data=[line.strip().split(",") for line in data]
    header=data[0]
    data=data[1:]
hypothesis=[]
for i in range(len(data)):
    if(data[i][-1]=="Yes"):
        if(hypothesis):
            for j in range(len(data[i])-1):
                if(hypothesis[j]!=data[i][j]):
                    hypothesis[j]="?"
        else:
            hypothesis=data[i][0:-1]
print(hypothesis)
