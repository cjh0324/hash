f1 = open("txt/Xeon_paddr_hex", "a")
f2 = open("txt/Xeon_slice", "a")

with open("txt/mapping_Xeon") as bookfile:
    for line in bookfile:
        fields = line.split("\t")
        f1.write(fields[0]+"\n")
        f2.write(fields[1])
    f1.close()
    f2.close()