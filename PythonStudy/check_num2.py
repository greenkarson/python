"""
输入一个正整数判断是否为素数
Version: 0.1
Author: Karson
"""

num = int(input("请输入一个正整数："))
if num > 1:
    for i in range(2, num):
        if num % i == 0:
            print("%d不是素数" % num)
            break
    else:
        print("%d是素数" % num)
    """
    这里要细细品味这段代码，else其实不是和if是一对，而是和for并排的，我们常见的是if…else…或者if…elif…else诸如此类，
    但其实for也可以和else搭配出现，在这段代码里，当某一次遍历结果余数为0后，break生效，那循环就结束了，
    那与之成对出现的else代码也就不执行了；当所有遍历结束后没有一次余数为0，那该循环就转到else开始执行，打印输出“该数为质数”。
    """
else:
    print("%d不是素数" % num)