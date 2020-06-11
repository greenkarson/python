dst_path = "/Users/karson/Downloads/Dataset/12"

check = open(f"{dst_path}/part.txt")
c = 0
for i, line in enumerate(check):
    c += 1
print(i)
