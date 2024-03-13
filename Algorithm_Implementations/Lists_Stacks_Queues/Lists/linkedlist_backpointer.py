from dataclasses import dataclass


@dataclass
class LLNode:
    val: int
    nxt: "LLNode | None" = None


class LinkedList:
    def __init__(self):
        self.lst = None
        self.last = None

    def append(self, d):
        if self.lst is None:
            self.lst = LLNode(d)
            self.last = self.lst
        else:
            self.last.nxt = LLNode(d)
            self.last = self.last.nxt

    def delete(self, d):
        if self.lst is not None:
            if self.lst.val == d:
                if self.lst.nxt is None:
                    self.lst = None
                    self.last = None
                else:
                    self.lst = self.lst.nxt
            else:
                pointer = self.lst
                while pointer.nxt is not None and pointer.nxt.val != d:
                    pointer = pointer.nxt
                if pointer.nxt is not None:
                    pointer.nxt = pointer.nxt.nxt
                    if pointer.nxt is None:
                        self.last = pointer


def test_linked_list():
    ll = LinkedList()

    # Initially, the list should be empty
    assert ll.lst is None
    assert ll.last is None

    # Append elements to the list
    ll.append(1)
    assert ll.lst.val == 1, "Failed to append 1"
    assert ll.last.val == 1, "Last reference incorrect after appending 1"

    ll.append(2)
    ll.append(3)
    assert ll.lst.val == 1 and ll.last.val == 3, "Appended elements incorrectly"

    # Delete the first element
    ll.delete(1)
    assert ll.lst.val == 2, "Failed to delete the first element correctly"

    # Delete a middle element
    ll.append(4)
    ll.delete(3)
    assert ll.lst.nxt.val == 4, "Failed to delete a middle element"

    # Delete the last element
    ll.delete(4)
    assert ll.last.val == 2, "Failed to delete the last element correctly"
    assert (
        ll.lst.nxt is None
    ), "Linked list nxt not correctly updated after deleting the last element"

    # Delete a non-existing element
    before_delete = ll.lst.val
    ll.delete(5)  # Attempt to delete a non-existent value
    assert (
        ll.lst.val == before_delete
    ), "List altered after attempting to delete a non-existent element"

    # Append and then delete until empty
    ll.append(6)
    ll.delete(2)
    ll.delete(6)
    assert (
        ll.lst is None and ll.last is None
    ), "List not empty after deleting all elements"

    print("All tests passed.")


test_linked_list()
