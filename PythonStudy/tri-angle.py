"""
判断输入的边长能否构成三角形，如果能则计算出三角形的周长和面积
Version: 0.1
Author: karson
"""
a = float(input("输入a:"))
b = float(input("输入b:"))
c = float(input("输入c:"))

if a + b > c and a + c > b and b + c > a:
    print("周长是：%f" % (a+b+c))
    p = a + b + c
    s = (p * (p - a) * (p - b) * (p - c)) ** 0.5
    print("面积是：%f" % s)