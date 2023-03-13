f2 = open("txt/sequence_idx", "w")

i = 0

with open("txt/data_Xeon.csv") as bookfile:
    for line in bookfile:
        if i % 32768 == 0:
            f2.write(line[:13]+"\n")
        i += 1
        
f2.close()