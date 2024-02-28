def twosum(lst) -> list[tuple[int, int]]:
    res = []
    cache = set()
    
    i = 0
    while i < len(lst):
        if -lst[i] in cache:
            res.append([lst[i],-lst[i]])
        cache.add(lst[i])
        i += 1
    
    return res


assert(twosum([1, 1, 1, 0])) == []
assert(twosum([-1, 1, 2, 4])) == [[1,-1]]
assert(twosum([0, 0, -2, 2])) == [[0,0],[2,-2]]
assert(twosum([1000, 1500, 500, -500])) == [[-500,500]]
print("All tests pass")
