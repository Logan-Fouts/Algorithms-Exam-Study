def twosum(lst) -> list[tuple[int, int]]:
    res = []
    lst = sorted(lst)
    frontPointer, endPointer = 0, len(lst) - 1

    while frontPointer < endPointer:
        currSum = lst[frontPointer] + lst[endPointer]

        if currSum == 0:
            res.append([lst[frontPointer], lst[endPointer]])
            frontPointer += 1
            endPointer -= 1
        elif currSum < 0:
            frontPointer += 1
        else:
            endPointer -= 1
    
    return res


assert(twosum([1, 1, 1, 0])) == []
assert(twosum([-1, 1, 2, 4])) == [[-1,1]]
assert(twosum([0, 0, -2, 2])) == [[-2,2],[0,0]]
assert(twosum([1000, 1500, 500, -500])) == [[-500,500]]
print("All tests pass")
