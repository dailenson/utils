import os
import shutil
import math
import numpy as np

class Node(object):
    def __init__(self,item):
        """
        param: self.elem 是结点的数据域
                self.lchild 是结点的左孩子
                self.rchild 是结点的右孩子
        """
        self.elem = item
        self.lchild = None
        self.rchild = None

class Tree(object):
    def __init__(self, item):
        self.root = Node(item)
        
    def create(self, stop_num):
        """
        param: stop_num是节点停止分裂的数量
        """
        queue = [self.root]
        
        while queue:
            """队列的弹出要加0,与栈相仿"""
            cur_node = queue.pop(0)
            temp = math.ceil(len(cur_node.elem)/2)
            
            if temp < stop_num: ### 当前节点数目不满足分裂要求
                continue
            
            if cur_node.lchild is None: 
                cur_node.lchild = Node(cur_node.elem[0:len(cur_node.elem)//2])
                queue.append(cur_node.lchild)

            """同理对右边的操作一样,还是手敲下吧"""
            if cur_node.rchild is None:
                cur_node.rchild = Node(cur_node.elem[len(cur_node.elem)//2:])
                queue.append(cur_node.rchild)
                
    def breadth_travel(self):
        """广度遍历与结点的添加非常相似,广度遍历不用插入结点了,在循环里面的条件和添加的相仿"""
        if self.root is None:
            print("the tree is null")
            return 
        
        queue = [self.root]
        labels = []
        while queue:
            cur_node = queue.pop(0)
            # 我们打印看看结点的遍历顺序对不对
            print(cur_node.elem)
            labels.append(cur_node.elem)
            if cur_node.lchild is not None:
                # 扔进队列循环
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)   
        return labels
        
def process_data(source_dir, dist_dir, items):
    for file_name in os.listdir(source_dir):     
        file_path = os.path.join(source_dir, file_name)
        with open(file_path,'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                _, label = [i for i in lines.split()]
                ##判断当前记录属于哪些partial数据集
                for i in range(len(items)):        
                    if int(label) in items[i]:
                        temp_name = str(i) + "_" + file_name
                        with open(os.path.join(dist_dir, temp_name), "a") as f:
                                f.write(lines)
        print(file_name)


if __name__ == "__main__":
    path = "/home/datasets/PACS"
    source_dir = os.path.join(path, "pacs_label/train")
    dist_name = "pacs_label/train"
    dist_dir = os.path.join(path, dist_name)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    root = np.arange(1, 8)
    new_tree = Tree(root)
    new_tree.create(2)
    items = new_tree.breadth_travel()
    process_data(source_dir, dist_dir, items)
