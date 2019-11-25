# * * * * * * * * * * * * * * * * * * *                      * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * Chapter One: Sorting * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * *                      * * * * * * * * * * * * * * * * * *

def less(a, i, j):
    return a[i] < a[j]


def exch(a, i, j):
    t = a[i]
    a[i] = a[j]
    a[j] = t


def is_sorted(a):
    for i in range(1, len(a)):
        if less(a, a[i], a[i-1]):
            print('not_sorted')
    print('sorted')


# def selection_sort(a):
#     for i in range(len(a)):
#         for j in range(i+1, len(a)):
#             if less(a, j, i):
#                 exch(a, i, j)


def selection_sort(a):
    for i in range(len(a)):
        min_num = i
        for j in range(i+1, len(a)):
            if less(a, j, min_num):
                min_num = j
        exch(a, min_num, i)


# def insertion_sort(a):
#     for i in range(len(a)):
#         while i > 0:
#             if less(a, i, i-1):
#                 exch(a, i, i-1)
#             i = i-1


def insertion_sort(a):
    for i in range(len(a)):
        while i > 0 and less(a, i, i-1):
            exch(a, i, i-1)
            i = i-1


# def shell_sort(a):  #Draft, but it works!
#     l = len(a)
#     h = 1
#     while h < l/3:
#         h = 3 * h + 1
#     while h > 0:
#         i = h
#         while i < l:
#             o = i
#             while i - h >= 0:
#                 if less(a, i, i-h):
#                     exch(a, i, i-h)
#                 i = i - h
#             i = o
#             i = i + 1
#         h = int(h/3)


def shell_sort(a):
    l = len(a)
    h = 1
    while h < l/3:
        h = 3 * h + 1
    while h > 0:
        for i in range(h, l):
            while i > 0 and less(a, i, i - h):
                exch(a, i, i - h)
                i = i - h
        h = int(h/3)


def merge(a, lo, mid, hi):
    i = lo
    j = mid + 1
    aux = list(a)
    for k in range(lo, hi+1):
        if i > mid:
            a[k] = aux[j]
            j += 1
        elif j > hi:
            a[k] = aux[i]
            i += 1
        elif less(aux, j, i):
            a[k] = aux[j]
            j += 1
        else:
            a[k] = aux[i]
            i += 1


def merge_sort(a, lo, hi):
    if hi <= lo:
        return
    mid = int(lo + (hi - lo) / 2)
    merge_sort(a, lo, mid)
    merge_sort(a, mid + 1, hi)
    merge(a, lo, mid, hi)


# def mergebu_sort(a):
#     lo = 0
#     size = 1
#     while size < len(a):
#         while lo < len(a) - size:
#             merge(a, lo, lo+size-1, min(lo+size+size-1, len(a) - 1))
#             lo += size + size
#         size = size + size


def partition(a, lo, hi):
    i = lo
    j = hi
    v = lo
    while True:
        while True:  # loop until find i that a[v]<a[i]
            if i == hi:
                break
            elif less(a, v, i):
                break
            i += 1
        while True:  # loop until find j that a[v]>a[j]
            if j == lo:
                break
            elif less(a, j, v):
                break
            j -= 1
        if i >= j:
            break
        exch(a, i, j)
    exch(a, j, lo)
    return j


def quick_sort(a, lo, hi):
    if hi <= lo:
        return
    k = partition(a, lo, hi)
    quick_sort(a, lo, k - 1)
    quick_sort(a, k + 1, hi)


def quick3way(a, lo, hi):
    if hi <= lo:
        return
    v = a[lo]
    gt = hi
    lt = lo
    i = lo + 1
    while i <= gt:
        if a[i] < v:
            exch(a, i, lt)
            i += 1
            lt += 1
        elif v < a[i]:
            exch(a, gt, i)
            gt -= 1
        else:
            i += 1
    quick3way(a, lo, lt - 1)
    quick3way(a, gt + 1, hi)


def sink(a, k, size):
    while 2 * k + 1 <= size - 1:
        j = 2 * k + 1
        if less(a, j, j + 1):
            j += 1
        if less(a, k, j):
            exch(a, j, k)
        k = j


def heapify(a):
    k = int(len(a) / 2)
    while k >= 0:
        sink(a, k, len(a) - 1)
        k -= 1


def heap_sort(a):
    l = len(a) - 1
    heapify(a)
    while l >= 0:
        exch(a, 0, l)
        l -= 1
        sink(a, 0, l)


# a = ['s', 'o', 'r', 't', 'e', 'x', 'a', 'm', 'p', 'l', 'e']
# heap_sort(a)
# print(a)


# import time
# import random
# a = b = test_data = list(range(50))
# random.shuffle(a)

# t1 = time.time()
# heap_sort(a)
# t2 = time.time()
# time_elapse1 = (t2 - t1)

# random.shuffle(b)
# t3 = time.time()
# shell_sort(b)
# t4 = time.time()
# time_elapse2 = (t4 - t3)

# ratio = time_elapse2 / time_elapse1
# print(ratio)


# * * * * * * * * * * * * * * * * *                        * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * Chapter Two: Searching * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * *                        * * * * * * * * * * * * * * * * * *

# from collections import Counter
# with open('/Users/frozen/Desktop/tale.txt') as t:
#     words_counter = Counter(t.read().split())
#     heap_sort(words_counter)
#     print(words_counter)
# for i in words_counter:
#     print(i, ':', words_counter[i])


def rank(a, k, l):
    hi = l - 1
    lo = 0
    while hi >= lo:
        mid = int(lo + (hi - lo) / 2)
        if str(k) < a[mid]:
            hi = mid - 1
        elif str(k) > a[mid]:
            lo = mid + 1
        else:
            return mid
    return lo


f = open("/Users/frozen/Desktop/amendments.txt", "r")
content = f.read()
f.close()
a = content.split()
size = len(a)
keys = [None] * size
vals = [0] * size
# print(keys)


def put(key, val, j):
    i = rank(keys, key, j)
    if keys[i] == key and i < j:
        vals[i] = val
        return
    while j > i:
        keys[j] = keys[j - 1]
        vals[j] = vals[j - 1]
        j -= 1
    vals[i] = val
    keys[i] = key


o = 0
for k in range(size):
    if a[k] in keys:
        put(a[k], vals[rank(keys, a[k], o)] + 1, o)
    else:
        put(a[k], 1, o)
        o += 1


frequency_counter = {}
i = 0
while i <= size and keys[i] is not None:
    frequency_counter[keys[i]] = vals[i]
    i += 1


# print(keys)
# print(vals)
# print(frequency_counter)


class TreeNode:

    def __init__(self, key, val, size=0, left=None, right=None, parent=None):
        self.key = key
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent
        self.size = size

    def hasleft(self):
        return self.left

    def hasright(self):
        return self.right


class BST:

    def __init__(self):
        self.root = None
        self.size = 0

    def sub_tree_size(self, currentnode):
        if currentnode is None:
            return 0
        else:
            return currentnode.size

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.val
            else:
                return None
        else:
            return None

    def _get(self, key, currentnode):
        if not currentnode:
            return None
        elif key < currentnode.key:
            return self._get(key, currentnode.left)
        elif key > currentnode.key:
            return self._get(key, currentnode.right)
        else:
            return currentnode

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False

    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val, 1)

    def _put(self, key, val, currentnode):
        if key < currentnode.key:
            if currentnode.hasleft():
                self._put(key, val, currentnode.left)
            else:
                currentnode.left = TreeNode(key, val, 1, parent=currentnode)
        elif key > currentnode.key:
            if currentnode.hasright():
                self._put(key, val, currentnode.right)
            else:
                currentnode.right = TreeNode(key, val, 1, parent=currentnode)
        elif key == currentnode.key:
            currentnode.val = val
        currentnode.size = self.sub_tree_size(currentnode.left) + self.sub_tree_size(currentnode.right) + 1
        self.size = self.root.size

    def __setitem__(self, k, v):
        self.put(k, v)

    def min(self, currentnode):
        if currentnode.left is None:
            return currentnode
        else:
            return self.min(currentnode.left)

    def floor(self, key, currentnode):
        if currentnode is None:
            return None
        if key == currentnode.key:
            return currentnode
        if key < currentnode.key:
            return self.floor(key, currentnode.left)
        a = self.floor(key, currentnode.right)
        if a is not None:
            return a
        else:
            return currentnode

    def select(self, k, currentnode):
        if currentnode is None:
            return None
        t = self.sub_tree_size(currentnode.left)
        if k < t:
            return self.select(k, currentnode.left)
        elif k > t:
            return self.select(k - t - 1, currentnode.right)
        else:
            return currentnode

    def rank(self, key, currentnode):
        if key is None:
            return None
        if key < currentnode.key:
            return self.rank(key, currentnode.left)
        elif key > currentnode.key:
            return self.sub_tree_size(currentnode.left) + 1 + self.sub_tree_size(currentnode.right)
        else:
            return self.sub_tree_size(currentnode)

    def deletemin(self, currentnode):
        if currentnode.left is None:
            return currentnode.right
        currentnode.left = self.deletemin(currentnode.left)
        currentnode.size = self.sub_tree_size(currentnode.left)+self.sub_tree_size(currentnode.right)+1
        self.size = self.root.size
        return currentnode

    def deletenode(self, key, currentnode):
        if currentnode is None:
            return None
        if key < currentnode.key:
            currentnode.left = self.deletenode(key, currentnode.left)
        elif key > currentnode.key:
            currentnode.right = self.deletenode(key, currentnode.right)
        else:
            if currentnode.right is None:
                return currentnode.left
            if currentnode.left is None:
                return currentnode.right
            t = currentnode
            currentnode = self.min(t.right)
            currentnode.right = self.deletemin(t.right)
            currentnode.left = t.left
        currentnode.size = self.sub_tree_size(currentnode.left)+self.sub_tree_size(currentnode.right)+1
        self.size = self.root.size
        return currentnode


class RedBlackNode(TreeNode):

    def __init__(self, key, val, color, size=0, left=None, right=None, parent=None):
        super().__init__(key, val, size, left, right, parent)
        self.color = color

    def is_red(self):
        if self.color is None:
            return False
        return self.color == 'Red'


class RedBlackBST(BST):

    def __init__(self):
        super().__init__()

    def is_red(self, node):
        if node is None:
            return False
        return node.color == 'Red'

    def rotate_left(self, node):
        x = node.right
        node.right = x.left
        x.left = node
        x.color = node.color
        node.color = 'Red'
        x.size = node.size
        node.size = self.sub_tree_size(node.left)+self.sub_tree_size(node.right)+1
        return x

    def rotate_right(self, node):
        x = node.left
        node.left = x.right
        x.right = node
        x.color = node.color
        node.color = 'Red'
        x.size = node.size
        node.size = self.sub_tree_size(node.left) + self.sub_tree_size(node.right) + 1
        return x

    def flip_color(self, node):
        node.color = 'Red'
        node.left.color = 'Black'
        node.right.color = 'Black'

    def put(self, key, val):
        self.root = self._put(self.root, key, val)
        self.root.color = 'Black'

    def _put(self, root, key, val):
        if root is None:
            return RedBlackNode(key=key, val=val, color='Red', size=1)
        if key < root.key:
            root.left = self._put(root.left, key, val)
        elif key > root.key:
            root.right = self._put(root.right, key, val)
        else:
            root.val = val
        if self.is_red(root.right) and not self.is_red(root.left):
            root = self.rotate_left(root)
        if self.is_red(root.left) and self.is_red(root.left.left):
            root = self.rotate_right(root)
        if self.is_red(root.left) and self.is_red(root.right):
            self.flip_color(root)
        root.size = self.sub_tree_size(root.left) + self.sub_tree_size(root.right) + 1
        self.size = self.root.size
        return root


# my_tree = RedBlackBST()
# # print(my_tree.__dict__)
# a = ['s', 'e', 'a', 'r', 'c', 'h', 'x', 'm', 'p', 'l']
# b = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# i = 0
# while i <= 9:
#     my_tree.put(a[i], b[i])
#     i += 1

# print(my_tree.root.left.right.key


# a = ['a', 'b', 'c']
#
# for i, j in enumerate(a):
#     a, b = i, j
#     print(a, b, 'k')


# j = 0
# print(my_tree.size)
# print(my_tree.deletenode('e', my_tree.root).key)
# my_tree.deletemin(my_tree.root)
# print(my_tree.size)
# print(my_tree.root.left.key)
# print(my_tree.select(9, my_tree.root).key)
# print(my_tree.sub_tree_size(my_tree.root))
# print(my_tree.floor('g', my_tree.root).key)
# while j <= 12:
#     print(my_tree[a[j]])
#     j += 1
# print(my_tree.sub_tree_size(my_tree.root.left.right))
# print(my_tree['e'])
# print(my_tree.root.hasleft())
# print(my_tree.root.key)
# print(my_tree._get('r', my_tree.root).right.key)
# print(my_tree.__dict__)
# print(help(BST))


# * * * * * * * * * * * * * * * * *                         * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * *  Chapter Three: Graph   * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * *                         * * * * * * * * * * * * * * * * * *


class Node:

    def __init__(self, item):
        self.item = item
        self.next = None


class Linklist:

    def __init__(self, first_node=None):
        self.first = first_node

    def __iter__(self):
        current = self.first
        while current is not None:
            yield current.item
            current = current.next

    def add(self, item):
        old_first = self.first
        self.first = Node(item)
        self.first.next = old_first

    def pop(self):
        item = self.first
        item.next = self.first
        return item


class Graph:

    def __init__(self, v, e):
        self.vertex = v
        self.edge = e
        self.adj = []
        for j in range(v):
            self.adj.append(Linklist(Node(j)))

    def add_edge(self, v, w):
        self.adj[v].add(w)
        self.adj[w].add(v)
        self.edge += 1


class DeepFirstSearch:

    def __init__(self, graph, start):
        self.marked = []
        self.edge_to = []
        for node in range(graph.vertex):
            self.marked.append(False)
            self.edge_to.append(None)
        self.count = 0
        self.start = start
        self.dfs(graph, start)

    def dfs(self, graph, v):
        self.marked[v] = True
        self.count += 1
        for i in graph.adj[v]:
            if self.marked[i] is not True:
                self.edge_to[i] = v
                self.dfs(graph, i)

    def path_to(self, v):
        path = Linklist()
        if self.marked[v] is not True:
            return False
        while v != self.start:
            path.add(v)
            v = self.edge_to[v]
        path.add(self.start)
        return path


g = Graph(6, 8)
g.add_edge(0,5)
g.add_edge(2,4)
g.add_edge(2,3)
g.add_edge(1,2)
g.add_edge(0,1)
g.add_edge(3,4)
g.add_edge(3,5)
g.add_edge(0,2)

# dfp = DeepFirstSearch(g, 0)
# print(dfp.edge_to[3])
# print(dfp.path_to(5).first.next.item)
# print(g.adj[2].first.next.next.item)
# a = Linklist()
# a.add(1)
# a.add(2)
# a.add(3)
# print(a.first.next.next.next)
# import numpy as np
# x = np.array([[1.0, -0.5], [-2.0, 3.0]])
# mask = (x <= 0)
# print(mask)
# out = x.copy()
# out[mask] = 0
# print(out)
a = [0,0,0]
b = [9,9,9]
a[b] = 0
print(a)
