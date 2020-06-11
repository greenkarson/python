class Node:
    def __init__(self,elem = -1,lchild = None,rchild = None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild


class Tree:
    def __init__(self):
        self.root = Node
        self.queue = []

    def add(self,elem):
        node = Node(elem)
        if self.root == None:
            self.root = node
            self.queue.append(self.root)
        else:
            tree_node = self.queue[0]
            if tree_node == None:
                tree_node.lchild = node
                self.queue.append(tree_node.lchild)
            else:
                tree_node.rchild = node
                self.queue.append(tree_node.rchild)
                self.queue.pop(0)

    # 前序遍历
    def pre_traverse(self,root):
        if root == None:
            return
        print(root.elem)
        self.pre_traverse(root.lchild)
        self.pre_traverse(root.rchild)
    # 中序遍历
    def mid_traverse(self,root):
        if root == None:
            return
        self.mid_traverse(root.lchild)
        print(root.elem)
        self.mid_traverse(root.rchild)
    # 后序遍历
    def after_traverse(self,root):
        if root == None:
            return
        self.after_traverse(root.lchild)
        self.after_traverse(root.rchild)
        print(root.elem)

item = range(6)
tree = Tree()
for i in item:
    tree.add(i)

tree.pre_traverse(tree.root)
print("===================")
tree.mid_traverse(tree.root)
print("===================")
tree.after_traverse(tree.root)