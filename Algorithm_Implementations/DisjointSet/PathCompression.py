# A Path Compression implementation

def init(n=20):
    return [i for i in range(n)]

def root(lst, a):
    while lst[a] != a:
        lst[a] = lst[lst[a]]
        a = lst[a]

    return a

def connected(lst, a, b):
    return root(lst, a) == root(lst, b)

def union(lst, a, b):
    root_a = root(lst, a)
    root_b = root(lst, b)

    lst[root_b] = root_a


lst = init(5)
union(lst, 0, 1)
union(lst, 0, 2)
union(lst, 3, 4)
union(lst, 0, 3)
print(lst)
print(root(lst, 4))
print("All tests passed.")
