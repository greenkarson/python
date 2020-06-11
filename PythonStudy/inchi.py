"""
英制转换公制
version: 0.1
author: karson
"""

value = float(input("请输入长度"))
unit = input("请输入单位")
if unit == "in":
    print("转换为厘米是%.1f" % (value*2.54))
elif unit == "cm":
    print("转换为英寸是%.1f" % (value * 0.3937))
else:
    print("请输入正确单位")