# A Path Compression implementation

def init(n=20):
    return [i for i in range(n)]

def connected(lst, a, b):
    return root(lst, a) == root(lst, b)

def union(lst, a, b):
    a_root = root(lst, a)
    b_root = root(lst, b)
    lst[b_root] = a_root

def root(lst, a):
    while lst[a] != a:
        lst[a] = lst[lst[a]]
        a = lst[a]
    return a



def test_init():
    lst = init(10)
    assert len(lst) == 10, "Length of list should be 10"
    assert all(x == i for i, x in enumerate(lst)), "List elements should be initialized to their indices"

test_init()

def test_root():
    lst = [0, 1, 2, 3, 4, 3, 3, 7, 8, 3]
    assert root(lst, 5) == 3, "Root of element 5 should be 3"
    assert root(lst, 9) == 3, "Root of element 9 should be 3"
    assert root(lst, 3) == 3, "Root of element 3 should be 3"

test_root()

def test_union():
    lst = init(10)
    union(lst, 0, 1)
    assert connected(lst, 0, 1), "0 and 1 should be connected"
    union(lst, 2, 3)
    assert connected(lst, 2, 3), "2 and 3 should be connected"
    union(lst, 0, 2)
    assert connected(lst, 0, 3), "0 and 3 should be connected after union"

test_union()

def test_connected():
    lst = init(10)
    union(lst, 0, 1)
    union(lst, 1, 2)
    assert connected(lst, 0, 2), "0 and 2 should be connected"
    assert not connected(lst, 0, 3), "0 and 3 should not be connected"

test_connected()
