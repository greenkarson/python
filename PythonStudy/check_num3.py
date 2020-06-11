"""
输入两个正整数计算它们的最大公约数和最小公倍数
Version: 0.1
Author: karson
"""

a = int(input("请输入一个数："))
b = int(input("请输入另外一个数："))
while a % b != 0:
    MOD = a % b
    a = b
    b = MOD

print("gcd(a,b)=", b)
