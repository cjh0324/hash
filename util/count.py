count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
limit = 0

with open("txt/mapping_Xeon") as bookfile:
    for line in bookfile:
        num = int(line[-3:-1])
        count[num] += 1
        limit += 1
        if (limit == 4096): 
            break

for i, count in enumerate(count):
    print("index: ", i, " ", "count: ", count)