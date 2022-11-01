f2 = open("txt/paddr_bin", "w")

scale = 16
num_of_bits = 34

with open("txt/paddr_hex") as bookfile:
    for line in bookfile:
        my_hexdata = line
        f2.write(bin(int(my_hexdata, scale))[2:].zfill(num_of_bits)+"\n")
    print(bin(int(my_hexdata, scale)))
    f2.close()