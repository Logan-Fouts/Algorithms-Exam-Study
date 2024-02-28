# A Weighted Quick Union implementation

def init(n=20):
    return [i for i in range(n)], [1]*n

def root(lst, a):
    while lst[a] != a:
        a = lst[a]
    return a

def connected(lst, a, b):
    return root(lst, a) == root(lst, b)

def union(sz, lst, a, b):
    root_a = root(lst, a)
    root_b = root(lst, b)

    if sz[root_a] < sz[root_b]:
        lst[root_a] = root_b
        sz[root_b] += sz[root_a]
    else:
        lst[root_b] = root_a
        sz[root_a] += sz[root_b]


lst, sz = init(5)
union(sz, lst, 0, 1)
union(sz, lst, 0, 2)
union(sz, lst, 3, 4)
union(sz, lst, 0, 3)
print(lst, sz)
print(root(lst, 4))
