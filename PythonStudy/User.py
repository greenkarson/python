"""
用户身份验证
version = 0.1
author = karson
"""

user = input("请输入用户名")
password = input("请输入密码")
if user == "admin" and password == "123456":
    print("用户名正确密码正确")
else:
    print("用户名密码错误")