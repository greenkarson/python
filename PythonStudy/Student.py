"""
定义一个类，并对类进行初始化操作
设计类的行为并调用执行

"""

class Student:
    def __init__(self, name, age):
        self.name = name   # 类中所定义的变量为属性
        self.age = age

    def get_study(self, course_name):  # 类中所定义函数为方法或称成员函数 必须带有self参数
        print("%s正在学习%s课程" % (self.name, course_name))


stu = Student("karson", 28)

stu.get_study("python")
