"""
分段函数求解
        3x - 5  (x > 1)
f(x) =  x + 2   (-1 <= x <= 1)
        5x + 3  (x < -1)
version = 0.1
author = karson
"""

x = float(input("x=:"))

if x > 1:
    print("y=%.1f" % (3*x-5))
elif x < -1:
    print("y=%.1f" % (5*x+3*x))
else:
    print("y=%.1f" % (x+2))