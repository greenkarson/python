"""
百分制成绩转换
version: 0.1
author: karson
"""
grade = float(input("请输入分数："))

if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
elif grade >= 70:
    print("C")
elif grade >= 60:
    print("D")
else:
    print("E")
