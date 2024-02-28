# A Weighted Quick Union implementation

def init(n=20):
    return [i for i in range(n)], [1]*n

def connected(lst, a, b):
    return root(lst, a) == root(lst, b)

def root(lst, a):
    while lst[a] != a:
        a = lst[a]
    return a

def union(sz, lst, a, b):
    a_id = root(lst, a)
    b_id = root(lst, b)

    if sz[a_id] < sz[b_id]:
        lst[a_id] = b_id
        sz[b_id] += sz[a_id]
    else:
        lst[b_id] = a_id
        sz[a_id] += sz[b_id]

def test_init():
    lst, sz = init(10)
    assert len(lst) == 10, "Length of list should be 10"
    assert len(sz) == 10, "Length of size array should be 10"
    assert all(x == i for i, x in enumerate(lst)), "List elements should be initialized to their indices"
    assert all(x == 1 for x in sz), "Size array elements should be initialized to 1"

test_init()

def test_root():
    lst = [0, 1, 2, 3, 4, 3, 3, 7, 8, 3]
    assert root(lst, 5) == 3, "Root of element 5 should be 3"
    assert root(lst, 9) == 3, "Root of element 9 should be 3"
    assert root(lst, 3) == 3, "Root of element 3 should be 3"

test_root()

def test_union():
    lst, sz = init(10)
    union(sz, lst, 0, 1)
    assert connected(lst, 0, 1), "0 and 1 should be connected"
    union(sz, lst, 2, 3)
    assert connected(lst, 2, 3), "2 and 3 should be connected"
    union(sz, lst, 0, 2)
    assert connected(lst, 0, 3), "0 and 3 should be connected after union"

test_union()

def test_connected():
    lst, sz = init(10)
    union(sz, lst, 0, 1)
    union(sz, lst, 1, 2)
    assert connected(lst, 0, 2), "0 and 2 should be connected"
    assert not connected(lst, 0, 3), "0 and 3 should not be connected"

test_connected()
