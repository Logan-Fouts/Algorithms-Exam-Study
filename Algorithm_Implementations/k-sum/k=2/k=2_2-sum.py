def twosum(lst) -> list[tuple[int, int]]:
    res = []

    for i in range(len(lst)):
        for j in range(len(lst)):
            if lst[i] + lst[j] == 0 and i != j:
                res.append([lst[i], lst[j]])

    return res



assert(twosum([1, 1, 1, 0])) == []
assert(twosum([-1, 1, 2, 4])) == [[-1,1],[1,-1]]
assert(twosum([0, 0, -2, 2])) == [[0,0],[0,0],[-2,2],[2,-2]]
assert(twosum([1000, 1500, 500, -500])) == [[500,-500],[-500,500]]
print("All tests pass")
