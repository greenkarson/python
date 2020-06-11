class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class Linkedlist:
    def __init__(self):
        self.head = None

    # is_empty() 链表是否为空
    def is_empty(self):
        return self.head is None
    # length() 链表长度
    def lenth(self):
        cur = self.head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    # items() 获取链表数据迭代器
    def items(self):
        cur = self.head
        while cur is not None:
            yield cur.data
            cur = cur.next

    def add(self, item):
        """向链表头部添加元素"""
        node = Node(item)
        # 新结点指针指向原头部结点
        node.next = self.head
        # 头部结点指针修改为新结点
        self.head = node

    # append(data) 链表尾部添加元素
    def append(self,data):
        new_node = Node(data)
        if self.is_empty():
            self.head = new_node
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            cur.next = new_node

    # insert(pos, item) 指定位置添加元素
    def insert(self,index,data):
        if index <= 0:
            self.add(data)
        elif index > (self.lenth()-1):
            self.append(data)
        else:
            node = Node(data)
            cur = self.head
            for i in range(index-1):
                cur = cur.next
            node.next = cur.next
            cur.next =node

    # remove(item) 删除节点
    def remove(self,data):
        cur = self.head
        pre = None
        while cur is not None:
            if cur.data == data:
                if not pre:
                    self.head = cur.next
                else:
                    pre.next = cur.next
                return True
            else:
                pre = cur
                cur = cur.next
    # find(item) 查找节点是否存在
    def find(self, item):
        """查找元素是否存在"""
        return item in self.items()

if __name__ == '__main__':
    link_list = Linkedlist()
    # 向链表尾部添加数据
    for i in range(10):
        link_list.append(i)
    # 链表数据插入数据
    link_list.insert(3, 9)

    # 向头部添加数据
    link_list.add(6)
    # 遍历链表数据
    for i in link_list.items():
        print(i, end='\t')
     # 删除链表数据
    link_list.remove(9)
    link_list.remove(6)
    print('\n', list(link_list.items()))

    # 查找链表数据
    print(link_list.find(4))