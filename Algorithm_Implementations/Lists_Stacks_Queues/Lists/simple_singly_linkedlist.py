from dataclasses import dataclass


@dataclass
class LLNode:
    val: int
    nxt: "LLNode | None" = None


class LinkedList:
    def __init__(self):
        self.lst = None

    def append(self, d):
        if self.lst is None:
            self.lst = LLNode(d)
        else:
            pointer = self.lst
            while pointer.nxt is not None:
                pointer = pointer.nxt
            pointer.nxt = LLNode(d)

    def delete(self, d):
        if self.lst is not None:
            if self.lst.val == d:
                self.lst = self.lst.nxt
            else:
                pointer = self.lst
                while pointer.nxt is not None and pointer.nxt.val != d:
                    pointer = pointer.nxt
                if pointer.nxt is not None:
                    pointer.nxt = pointer.nxt.nxt


def test_linked_list():
    ll = LinkedList()

    # Test appending to an empty list
    ll.append(10)
    assert ll.lst.val == 10, "Failed to append to an empty list"

    # Test appending multiple items
    ll.append(20)
    ll.append(30)
    assert (
        ll.lst.nxt.val == 20 and ll.lst.nxt.nxt.val == 30
    ), "Failed to append multiple items"

    # Test deleting a non-existent item (should not change the list)
    ll.delete(40)
    assert (
        ll.lst.val == 10 and ll.lst.nxt.val == 20 and ll.lst.nxt.nxt.val == 30
    ), "List changed after deleting non-existent item"

    # Test deleting the first item
    ll.delete(10)
    assert ll.lst.val == 20, "Failed to delete the first item"

    # Test deleting a middle item
    ll.append(40)
    ll.delete(30)
    assert ll.lst.val == 20 and ll.lst.nxt.val == 40, "Failed to delete a middle item"

    # Test deleting the last item
    ll.delete(40)
    assert ll.lst.val == 20 and ll.lst.nxt is None, "Failed to delete the last item"

    # Test deleting the only item in the list
    ll.delete(20)
    assert ll.lst is None, "Failed to delete the only item in the list"

    print("All tests passed!")


test_linked_list()
