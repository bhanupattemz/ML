import csv
data=[]
header=[]
with open(r"c:\Users\bhanu\OneDrive\Desktop\@jntua\ML_lab\Prac\fruit.csv", "r") as f:
    data=f.readlines()
    header=data[0].strip().split(",")
    data=[line.strip().split(",") for line in data[1:]]
y_valuse=0

for i in range(len(data)):
    if(data[i][-1]=="Yes"):
        y_valuse+=1
prob_yes=y_valuse/len(data)
prob_No=(len(data)-y_valuse)/len(data)

prob={}
for i in range(len(header)-1):
    prob[header[i]]={}
    valuse={}
    for j in range(len(data)):
        if(data[j][i] not in valuse):
            valuse[data[j][i]]={
                "Yes":0,
                "No":0
            }
        if(data[j][-1]=="Yes"):
            valuse[data[j][i]]["Yes"]+=1
        else:
            valuse[data[j][i]]["No"]+=1
    for key,val in valuse.items():
        prob[header[i]][key]={
            "Yes":(val["Yes"]/(val["Yes"]+val["No"])),
            "No":(val["No"]/(val["Yes"]+val["No"]))
        }

input={
    "Color":"Green",
    "Shape":"Oval",
    "Size":"Medium",
    "Texture":"Smooth",
    "Taste":"Sweet",
    "isInsideWhite":"No",
    "HasSeeds":"Yes",
    "Sugar":"Low",
    "Calories":"Medium"
}

for attr,val in prob.items():
    prob_yes*=val[input[attr]]["Yes"]
    prob_No*=val[input[attr]]["No"]
    
  
prob_No=prob_No/(prob_No+prob_yes)
prob_yes=prob_yes/(prob_No+prob_yes)
print(f"Probility of Yes :{prob_yes}\nProbility of No :{prob_No}")
if(prob_yes>prob_No):
    print("Yes")
else:
    print("No")

