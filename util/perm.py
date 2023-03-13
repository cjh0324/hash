perm = 0b000000000000

cmp1_arr = []
cmp2_arr = []

with open("txt/cmp1") as book1:
    for line1 in book1:
        cmp1_arr.append(int(line1[13:]))

with open("txt/cmp2") as book2:
    for line2 in book2:
        cmp2_arr.append(int(line2[13:]))
            
for perm in range(0,4096):
    if (perm >= 1024 and perm < 2048):
        perm += 1
        continue
    print(perm)
    for idx in range(0, 4096):
        idx_perm = idx ^ perm
        if (cmp1_arr[idx] != cmp2_arr[idx_perm]) :
            perm += 1
            break
        idx += 1
    if (idx == 4096):
        break

print('0b'+ bin(perm)[2:].zfill(12))