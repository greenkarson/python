"""
输入一个年份判断是否是闰年
version:0.1
author:karson
"""

year = int(input("请输入一个年份："))
is_leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0
print(is_leap)

