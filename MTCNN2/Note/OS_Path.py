import os,torch

path1 = "home"
path2 = "karson"
path3 = "work"
PATH = os.path.join(path1,path2,path3)
PATH2 = path1 + path2 + path3
print(PATH,PATH2)

PATH_cebela = "../cebela/48/positive.txt"
file = open(PATH_cebela,"r")

print(torch.randn(3,14,14) / 255.-0.5)

try:
    # 逐行读取
    text_lines = file.readlines()
    # 读入首行
    # text_lines = file.readline()
    strs = text_lines[0].strip().split(" ")
    print(torch.tensor([int(strs[1])]))
    print(text_lines)


    # text_lines = file.readlines()
    # print(type(text_lines), text_lines)
    # for line in text_lines:
    #     print(type(line), line)
finally:
    file.close()
