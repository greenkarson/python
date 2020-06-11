class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class Linkedlist:
    def __init__(self):
        self.head = None

    def is_empty(self):
        if self.head == Node:
            return self.head is None

    def lenth(self):
        cur = self.head
        count = 0
        while cur is not None:
            count = +1
            cur = self.next
        return count

    def item(self):
        cur = self.head
        while cur is not None:
            yield cur.data
            cur = cur.next

    def add(self,data):
        node = Node(data)
        if self.head == None:
            self.head = node
        else:
            node.next = self.head
            self.head = node

    def append(self,data):
        node = Node(data)
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur is not None:
                cur = cur.next
            cur.next = node

    def insert(self,index,data):
        node = Node(data)
        if index <= 0 :
            self.add(data)
        elif index > (self.lenth()-1):
            self.append(data)
        else:
            cur = self.head
            for i in range(index-1):
                cur =cur.next
            node.next = cur.next
            cur.next = node

    def delete(self,data):
        cur = self.head
        pre = None

