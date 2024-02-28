def twosum(lst, target) -> list[tuple[int, int]]:
    res = []
    lst = sorted(lst)
    frontPointer, endPointer = 0, len(lst) - 1

    while frontPointer < endPointer:
        currSum = lst[frontPointer] + lst[endPointer]

        if currSum == target:
            res.append([lst[frontPointer], lst[endPointer]])
            frontPointer += 1
            endPointer -= 1
        elif currSum < target:
            frontPointer += 1
        else:
            endPointer -= 1
    
    return res

def threesum(lst):
    res = []

    for i in range(len(lst)):
        twosum_res = twosum(lst[:i] + lst[i+1:], -lst[i])
        if twosum_res != []:
            res.append([lst[i],twosum_res[0][0],twosum_res[0][1]])

    return res


assert(threesum([1, 1, 1, 0])) == []
assert(threesum([-1, 1, 2, 4])) == []
assert(threesum([0, 0, -2, 2])) == [[0, -2, 2], [0, -2, 2], [-2, 0, 2], [2, -2, 0]]
assert(threesum([1000, 1500, 500, -500])) == []
print("All tests pass")
