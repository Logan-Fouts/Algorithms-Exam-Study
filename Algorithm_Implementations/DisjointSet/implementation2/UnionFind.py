# A Basic Union Find/Quick Find implementation

def init(n=20):
    return [i for i in range(n)] 

def connected(lst, a, b):
    return lst[a] == lst[b]

def union(lst, a, b):
    a_id = lst[a]
    b_id = lst[b]
    
    for i in range(len(lst)):
        if lst[i] == b_id:
            lst[i] = a_id

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
