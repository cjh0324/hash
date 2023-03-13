f2 = open("txt/cmp1", "w")

idx = 10715136
i = 0

with open("txt/data_Xeon.csv") as bookfile:
    for line in bookfile:
        if i > idx-1:
            f2.write(line[16:28]+","+line[35:])
        i += 1
        if (i == idx + 4096):
            break
        
f2.close()