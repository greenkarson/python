def factorial(num):
    result = 1
    for n in range(1, num + 1):
        result *= n
    return result


m = int(input("请输入一个数："))

print("%d这个数阶乘是%d" % (m, factorial(m)))
