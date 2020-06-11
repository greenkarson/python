"""
用for循环实现遍历列表、元祖、set集合、字典

Version: 0.1
Author: karson
"""
a = [1, 2, 3, 4, 5, 6]
for i in a:
    print("i=", i)

b = (6, 7, 8, 9, 10)
for x in b:
    print("x=", x)

c = {11, 12, 13}
for y in c:
    print("y=", y)

my_dic = {'python教程': "http://c.biancheng.net/python/",
          'shell教程': "http://c.biancheng.net/shell/",
          'java教程': "http://c.biancheng.net/java/"}

for d in my_dic.items():
    print("d", d)
"""
在使用 for 循环遍历字典时，经常会用到和字典相关的 3 个方法，即 items()、keys()、values()
当然，如果使用 for 循环直接遍历字典，则迭代变量会被先后赋值为每个键值对中的键。
"""