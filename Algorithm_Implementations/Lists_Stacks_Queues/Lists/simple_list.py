import numpy as np


class SimpleList:
    def __init__(self):
        self.lst = np.empty(10, dtype=np.int64)
        self.cnt = 0

    def append(self, d):
        if self.cnt < len(self.lst):
            self.lst[self.cnt] = d
            self.cnt += 1
        else:
            tmp = np.empty(self.cnt * 2, dtype=np.int64)
            for i, num in enumerate(self.lst):
                tmp[i] = num
            self.lst = tmp
            self.lst[self.cnt] = d
            self.cnt += 1

    def remove(self, d):
        found = False
        for i in range(self.cnt):
            if self.lst[i] == d:
                found = True
            if found and i + 1 < self.cnt:
                self.lst[i] = self.lst[i + 1]
        if found:
            self.cnt -= 1
            self.lst[self.cnt] = 0  # Reset the last element to 0 or a default value


sl = SimpleList()

# Test appending and count
sl.append(1)
sl.append(2)
assert sl.cnt == 2, "Append failed"

# Test removal of an existing item
sl.remove(1)
assert sl.cnt == 1, "Remove failed"
assert sl.lst[0] == 2, "List content incorrect after removal"

# Test removal of a non-existing item
initial_cnt = sl.cnt
sl.remove(3)  # Try to remove an item that's not in the list
assert sl.cnt == initial_cnt, "Remove non-existing item failed"

# Test append until full
for i in range(3, 12):  # Start from 3 to attempt exceeding capacity
    sl.append(i)
assert sl.cnt == 10, "Append or count failed when near capacity"

print("All tests passed")
