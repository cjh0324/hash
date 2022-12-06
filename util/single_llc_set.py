f2 = open("txt/single_llc_set_false.csv", "w")

bits = 34
llc_set_bit = 6 # Should be between 6 ~ 16

with open("txt/data_Xeon.csv") as bookfile:
    for line in bookfile:
        read = 1
        for i in range(6, 15):
            if (line[34 - 1 - i] != '0'): 
                read = 0
        if (read == 1) :
            f2.write(line)
        read = 1
        
f2.close()