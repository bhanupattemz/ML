import csv
data=[]
with open(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Prac\fruit.csv", "r") as f:
    data=f.readlines()
    data=[line.strip().split(",") for line in data]
hypo=[]
gHypo=[["?" for i in range(len(data[0]))]]*len(data)
for i in range(len(data)):
    if data[i][-1]=="Yes":
        hypo=data[1][0:-1]
        break
for i in range(i+1,len(data)):
    if data[i][-1]=="Yes":
        for j in range(len(hypo)):
            if data[i][j]!=hypo[j]:
                hypo[j]="?"
    else:
        for j in range(len(hypo)):
            if data[i][j]!=hypo[j]:
                gHypo[j][j]=hypo[j]
print(hypo)
print(gHypo)