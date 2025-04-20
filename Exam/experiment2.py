with open("c:/Users/bhanu/OneDrive/Desktop/@jntua/ML_lab/Exam/example.csv","r") as file:
    data=file.readlines()
    data=[line.strip().split(",") for line in data]
    header=data[0]
    data=data[1:]

g=[["?"]*(len(header)-1) for _ in range(len(header)-1)]
s=[None]*(len(header)-1)

if(data[0][-1]=="yes"):
    s=data[0][0:-1]
    for i in range(len(data)):
        if(data[i][-1]=="yes"):
            for j in range(len(data[i])-1):
                if(data[i][j]!=s[j]):
                    s[j]="?"
        else:
            for j in range(len(data[i])-1):
                if(data[i][j]!="?" and data[i][j]!=s[j]):
                    g[j][j]=s[j]
    print(f"general: {g}")
    print(f"Specific: {s}")

else:
    print("First instance must be positive.")
