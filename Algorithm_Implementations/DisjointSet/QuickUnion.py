# A Quick Union implementation
 
def init(n=20):
    return [i for i in range(n)] 

def connected(lst, a, b):
    return root(lst, a) == root(lst, b)

def root(lst, a):
    while lst[a] != a:
        a = lst[a]
    return a

def union(lst, a, b):
    root_a = root(lst, a)
    root_b = root(lst, b)
    lst[root_b] = root_a

lst = init()
assert not connected(lst, 0, 1), "Error: 0 and 1 should not be connected initially."
union(lst, 0, 1)
assert connected(lst, 0, 1), "Error: 0 and 1 should be connected after union."
union(lst, 1, 2)
assert connected(lst, 0, 2), "Error: 0 and 2 should be connected through 1."
union(lst, 2, 3)
union(lst, 3, 4)
assert connected(lst, 0, 4), "Error: 0 through 4 should all be connected."
assert not connected(lst, 0, 5), "Error: 0 and 5 should not be connected without a union operation."
print("All tests passed.")
