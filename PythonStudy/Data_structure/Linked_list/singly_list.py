class Node:
    def __init__(self,data):
        self.data = data
        self._next = None

    # def __repr__(self):
    #     return f"Node({self.data})"

class Linkedlist:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head is None

    def insert_head(self,data):
        new_node = Node(data)
        new_node._next = self.head
        self.head = new_node

    def insert_tail(self, data):
        if self.head is None:
            self.insert_head(data)  # if this is first node, call insert_head
        else:
            temp = self.head
            while temp.next:  # traverse to last node
                temp = temp._next
            temp.next = Node(data)  # create node & link to tail


    def print_list(self):  # print every node data
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp._next

    def items(self):
        """遍历链表"""
        # 获取head指针
        cur = self.head
        # 循环遍历
        while cur is not None:
            # 返回生成器
            yield cur.item
            # 指针下移
            cur = cur.next

if __name__ == '__main__':
    A = Linkedlist()
    A.insert_head(5)
    A.insert_head(5)

    A.print_list()
