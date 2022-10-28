f1 = open("txt/paddr", "a")
f2 = open("txt/slice", "a")

with open("mapping") as bookfile:
    for line in bookfile:
        fields = line.split("\t")
        f1.write(fields[0]+"\n")
        f2.write(fields[1])
    f1.close()
    f2.close()