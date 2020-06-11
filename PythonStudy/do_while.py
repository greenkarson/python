"""
猜数字游戏
计算机出一个1~50之间的随机数由人来猜
计算机根据人猜的数字分别给出提示大一点/小一点/猜对了

Version: 0.1
Author: karson
"""
import random
answer = random.randint(1, 50)
counter = 0
while True:
    counter += 1
    number = int(input("请输入一个数字"))
    if answer > number:
        print("大一点")
    elif answer < number:
         print("小一点")
    else:
        print("恭喜你猜对了")
        break  # 终止本次循环，while循环必须有

print("总共猜了%d次" % counter)
if counter > 7:
    print("您的智商不足")

