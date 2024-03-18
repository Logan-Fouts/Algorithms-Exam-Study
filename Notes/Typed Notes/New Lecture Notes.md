## Lecture 1 (Intro and Union Find) Ch. 8.1 - 8.5, 8.7

- The design of an algorithm or a data structure is about creating a solution to a problem.
- "Code is less important than the data structures and their relationships." - Linus Torvalds

### Connections

- Equivalence relation means: 
  - Reflective, a is connected to a.
  - Symmetric, if a is connected to b, then b is connected to a.
  - Transitive, if a is connected to b and b is connected to c then a is connected to c.
- Electrical connectivity, where all connections are by metal wires, is an equivalence relation
- A ==connected component ==is a maximal set of objects that are mutually connected. 
  - ![[Pasted image 20240221154419 1.png]]

### Disjoint Set

This is an efficient data structure to solve the equivalence problem.

#### Union-Find

- For quick find a sequence of N − 1 unions (the maximum, since then everything is in one set) would take Θ($N^2$) time.

Two main methods:

- Connected...

```python
# Simly check if they have the same id.
def connected(self, a:int, b:int) -> bool:
	return self.d[a] == self.d[b]

Time complexity:
Ω(1)
O(1)
Θ(1) 
```

- Union...

```python
# Check all nodes, if it has the id of the given (a) set it to the new id (b).
def union (self:UnionFind, a:int, b:int) -> None:
	a_id = self.d[a]
	b_id = self.d[b]
	
	for i, v in enumerate(self.d):
		if v == a_id:
			self.d[i] = b_id

Time complexity:
Ω(n)
O(n)
Θ(n) 
```

#### Quick-Union

- Each component is represented as a tree.
- Same root = same component.
- When adding an object to a component, its root is connected to the root of the other component.
- Example: 
  - ![[Pasted image 20240221161447.png]] Three Main methods:
- Root...

```python
# Keep checking the parent of the given element until the parent is itself.
def root(d:list[int], a:int) -> int:
	while a != d[a]:
		a = d[a]
	return a

Time complexity:
Ω(1)
O(n)
Cant really say average time complexity
```

- Connected...

```python
# Uses the root method to check if the nodes have the same root.
def connected(d:list[int], a:int, b:int) -> bool:
	return root(d, a) == root(d, b)

Time complexity:
Ω(2) = Ω(1)
O(2n) = O(n)
Cant really say average time complexity
```

- Union...

```python
# Get the root of each node and set the parent of the root of node a to be the root of node b.
def union(d:list[int], a:int, b:int) -> None:
	ra = root(d, a)
	rb = root(d, b)
	d[ra] = rb

Time complexity:
Ω(2) = Ω(1)
O(2n) = O(n)
Cant really say average time complexity
```

#### Weighted-Quick-Union (Also Called Union-By-Size)

- Attach the smaller tree to the root of the larger tree.
- If we also keep track of the size of each equivalence class, and when performing unions we change the name of the smaller equivalence class to the larger, then the total time spent for N − 1 merges is O(N log N). Three Main methods:
- Root...

```python
# Keep checking the parent of the given element until the parent is itself.
def root(self, a:int) -> int:
	while a != self.d[a]:
		a = self.d[a]
	return a

Time complexity:
Ω(1)
O(n)
Cant really say average time complexity
```

- Connected...

```python
# Uses the root method to check if the nodes have the same root.
def connected(self, a:int, b:int) -> bool:
	return self.root(a) == self.root(b)

Time complexity:
Ω(2) = Ω(1)
O(2n) = O(n)
Cant really say average time complexity
```

- Union...

```python
# Get the root of each node and then attach the smaller tree to the root of the larger tree.
def union(self:WQUnionFind, a:int, b:int) -> None:
	ra = self.root(a)
	rb = self.root(b)
	
	if self.sz[ra] < self.sz[rb]:
		self.d[ra] = rb
		self.sz[rb] += self.sz[ra]
	else:
		self.d[rb] = ra
		self.sz[ra] += self.sz[rb]

Time complexity:
Ω(2) = Ω(1)
O(2n) = O(n)
Cant really say average time complexity
```

#### Path-Compression & Weighted-Quick-Union

- The time to find the root depends on the height of the tree.
- Make trees flatter but wider.
- While we are looking for the root drag sub-trees up and attach them to the root.
- Example: 
  - ![[Pasted image 20240221170018.png]] Three Main methods:
- Root...

```python
# Keep checking the parent of the given element and setting the root of a to the current node until the parent is itself.
def root(self:WQUnionFind, a:int) -> int:
	while a != self.d[a]:
		self.d[a] = self.d[self.d[a]]
		a = self.d[a]
	return a

Time complexity:
Ω(1)
O(n)
Cant really say average time complexity
```

- Connected...

```python
# Uses the root method to check if the nodes have the same root.
def connected(d:list[int], a:int, b:int) -> bool:
	return root(d, a) == root(d, b)

Time complexity:
Ω(2) = Ω(1)
O(2n) = O(n)
Cant really say average time complexity
```

- Union...

```python
# Get the root of each node and then attach the smaller tree to the root of the larger tree.
def union(d:list[int], a:int, b:int) -> None:
	ra = root(d, a)
	rb = root(d, b)
	
	if self.sz[ra] < self.sz[rb]:
		self.d[ra] = rb
		self.sz[rb] += self.sz[ra]
	else:
		self.d[rb] = ra
		self.sz[ra] += self.sz[rb]

Time complexity:
Ω(2) = Ω(1)
O(2n) = O(n)
Cant really say average time complexity
```
## Lecture 2 (Algorithm Analysis) Ch. 2
### Linear Growth
- A dependent variable changes at a constant rate with respect to an independent variable
	- $y=mx+b$
	- m is the slope
	- b is the y-intercept
	- rise over run
### Exponential Growth
- No longer changing by a fixed amount
- $y=4(1.05)^x$
- $4(1.05)^x$ - This part describes the rate or pace of the change at a point x
### Calculating Slope

| n   | exctime |
| --- | ------- |
| 5   | .11     |
| 48  | .43     |
| 124 | .98     |
| 172 | 1.35    |
#### Use the coordinates to find the slope
- Use the math equations we calculate in order to estimate performance.
##### Normal
$$
\begin{flalign}
\frac{0.43-11}{48-5}=\frac{0.32}{43}=0.007 &&\\\\
\frac{1.35-0.48}{172-55}=\frac{0.87}{117}=0.007 &&
\end{flalign}
$$
##### log log Space
$$
\begin{flalign}
log_2 time(x)=b*log_2x+c&&\\\\
slope (b)=\frac{log_2y_1-log_2y_0}{log_2x_1-log_2x_0}&&\\\\
c=log_2time(x)-b*log_2x&&
\end{flalign}
$$
- ![[Pasted image 20240226193339.png]]
- Power-law
$$
\begin{flalign}
a+x^b&&\\
a=2^c&&
\end{flalign}
$$
- So from the example above: $2^{-22.4}*x^{2.022}$
```run-python
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2**(-22.4) * x**2.022

# Generate x values
x_values = np.linspace(0.1, 10, 100)

y_values = f(x_values)

plt.plot(x_values, y_values, label=r'$2^{-22.4} \times x^{2.022}$')

plt.xlabel('x')
plt.ylabel('y')
plt.title('log log plot')

plt.legend()

plt.grid(True)
plt.show()
```

### k-sum
- Assume we have a list of n integers. How many sequences of k numbers in the list sum to 0?
#### k=2 (2sum)
##### k=2 (2sum)
```run-python
def twosum(l:list[int]) -> list[tuple[int, int]]:
	res = []
	for i, vi in enumerate(l):
		for j, vj in enumerate(l):
			if i == j:
				continue
			if vi + vj == 0:
				res.append((vi, vj))
	return res
	
print(twosum([2,0,-2,1]))

# Time complexity:
# Ω(n^2)
# O(n^2)
# Θ(n^2)
```
##### k=2 (2sum) with smart iterations
- Iterate in the inside loop from `i+1` instead of from `i`
- Does not really improve complexity. However, performance of course is better.
```run-python
def twosum_si(l:list[int]) -> list[tuple[int, int]]:
	res = []
	for i, vi in enumerate(l):
		for vj in l[i+1:]:
			if vi + vj == 0:
				res.append((vi, vj))
	return res
	
print(twosum([2,0,-2,1]))

# Time complexity:
# Ω(n^2)
# O(n^2)
# Θ(n^2)
```
##### k=2 (2sum) with sorting
- Sort the input
	- If too small, pick a larger number in the front
	- If too large, pick a smaller number in the end
- Faster but not quite linear since we **can** **not really sort in linear time** despite the graph looking almost linear.
```run-python
def twosum_p(l:list[int]) -> list[tuple[int, int]]:
	res = []
	s = sorted(l)
	fp, bp = 0, len(s) -1
	while fp< bp:
		p = s[fp] + s[bp]
		if p == 0:
			res.append((s[fp], s[bp]))
			fp += 1
		elif p < 0:
			fp += 1
		else:
			bp -= 1
	return res
	
print(twosum_p([2,0,-2,1]))

# Time complexity:
# Ω(nlogn)
# O(nlogn)
# Θ(nlogn)
```

##### k=2 (2sum) with caching
- Store numbers we have already seen
- If we have seen the current negative number, then it is a match
- Faster but **more memory**
```run-python
def twosum_c(l: list[int]) -> list[tuple[int, int]]:
    cache = set()  # Use a set to store numbers we've seen
    res = []
    for vi in l:
        if -vi in cache:
            res.append((vi, -vi))
        cache.add(vi)  # Add the current number to the set
    return res

print(twosum_c([2, 0, -2, 1]))

# Time complexity:
# Ω(n)
# O(n)
# Θ(n)
```
#### k=3 (3sum)
##### k=3 (3sum) using 2sum and a target value
- Modify 2sum to take a target value instead of just 0
- Iterate over the list and use 2sum to look for two number that sum to the target negative
```run-python
def two_sum_target(l: list[int], target: int) -> list[tuple[int, int]]:
    res = []
    seen = {}
    for num in l:
        complement = target - num
        if complement in seen:
            res.append((complement, num))
        seen[num] = True
    return res

def three_sum_using_twosum(l: list[int]) -> list[tuple[int, int, int]]:
    res = set()
    for i, num in enumerate(l):
        pairs = two_sum_target(l[i+1:], -num)
        for pair in pairs:
            triplet = tuple(sorted([num, pair[0], pair[1]]))
            res.add(triplet)
    return list(res)
    
print(three_sum_using_twosum([2, 0, -2, 1, -1, 4, -3, 3]))

# Time complexity:
# Ω(n^2)
# O(n^2)
# Θ(n^2)
```
##### k=3 (3sum) using 2sum with cache and a target value
```run-python
def two_sum_target(l: list[int], target: int) -> list[tuple[int, int]]:
    res = []
    seen = {}
    for num in l:
        complement = target - num
        if complement in seen:
            res.append((complement, num))
        seen[num] = True
    return res

def three_sum_with_cache(l: list[int]) -> list[tuple[int, int, int]]:
    res = set()  
    for i, num in enumerate(l):
        pairs = two_sum_target(l[i+1:], -num)
        for pair in pairs:
            triplet = tuple(sorted([num, pair[0], pair[1]]))
            res.add(triplet)
    return list(res)

print(three_sum_with_cache([2, 0, -2, 1, -1, 4, -3, 3]))

# Time complexity:
# Ω(n^2)
# O(n^2)
# Θ(n^2)
```
##### k=3 (3sum) using 2sum with pointers and a target value
```run-python
def two_sum_pointers(l: list[int], target: int) -> list[tuple[int, int]]:
    res = []
    l.sort()
    fp, bp = 0, len(l) - 1
    while fp < bp:
        sum = l[fp] + l[bp]
        if sum == target:
            res.append((l[fp], l[bp]))
            fp += 1
            bp -= 1
            while fp < bp and l[fp] == l[fp - 1]:
                fp += 1
            while fp < bp and l[bp] == l[bp + 1]:
                bp -= 1
        elif sum < target:
            fp += 1
        else:
            bp -= 1
    return res

def three_sum_pointers(l: list[int]) -> list[tuple[int, int, int]]:
    l.sort()
    res = set()
    for i in range(len(l) - 2):
        if i > 0 and l[i] == l[i-1]:
            continue
        pairs = two_sum_pointers(l[i+1:], -l[i])
        for pair in pairs:
            triplet = (l[i], pair[0], pair[1])
            res.add(triplet)
    return list(res)
    
print(three_sum_pointers([2, 0, -2, 1, -1, 4, -3, 3]))

# Time complexity:
# Ω(n^2)
# O(n^2)
# Θ(n^2)
```

### Mathematical Models
"Imagine you're baking cookies, and you want to figure out how much effort it's going to take. Even if you don't get super detailed, you can still get a rough idea by considering the main steps, like mixing dough and baking. In computing, we also like to estimate how much work a task needs. Instead of mixing and baking, we count basic steps like adding or multiplying numbers, or looking up information."
#### Simplification: A cost model
- To avoid doing frequency calculations like the image below we use some basic operation as a proxy for the running time:
	- ![[Pasted image 20240226201947.png]]
##### Tilde notation
- Estimate runtime or memory use as a function of input size N
	- As N grows, the lower order terms are negligible
	- And if N is small, we do not care
- So, $N^3+5*N^2+100*N+10987$ ~ $N^3$
- Why we do not care
	- $f(N)$ ~ $g(N)$ means $\lim_{N \to \infty}\frac{f(N)}{g(N)}=1$
	- Explanation: This mathematical expression describes a way to compare two functions, (f(N)) and (g(N)), as the number (N) gets very large (approaches infinity). When it says (f(N) sim g(N)), it means that the ratio of (f(N)) to (g(N)) approaches 1 as (N) gets larger and larger.
	- In simpler terms, think of (f(N)) and (g(N)) as two different ways to measure something as you keep increasing (N). As (N) gets really big, these two measurements are getting closer and closer to being the same. So, if you can imagine increasing (N) forever, the two functions would be virtually indistinguishable from each other in terms of their growth rates or final values.


### Classifying the order of growth

| **order** | **name**     | **description**    | **$T(2N)/T(N)$** |
| --------- | ------------ | ------------------ | ---------------- |
| $1$       | constant     | statement          | 1                |
| $logN$    | logarithmic  | divide in half     | ~1               |
| $N$       | linear       | loop               | 2                |
| $NlogN$   | linearithmic | divide and conquer | ~2               |
| $N^2$     | quadratic    | double loop        | 4                |
| $N^3$     | cubic        | triple loop        | 8                |
![[Pasted image 20240226204134.png]]
![[Pasted image 20240226204122.png]]
### Notation
- Asymptotic- this refers to the behavior of functions as the argument goes towards a limit, often infinity.

| **notation** | **provides**               | **example** | **meaning**  |
| ------------ | -------------------------- | ----------- | ------------ |
| Big Theta    | asymptotic order of growth | $Θ(N^2)$    | Average case |
| Big Oh       | $Θ(N^2)$ and smaller       | $O(n^2)$    | Worst case   |
| Big Omega    | $Θ(N^2)$ and larger        | $Ω(N^2)$    | Best case    |
### Math definitions
- $T(N)=O(f(N))$ if there are positive constants $c$ and $n_0$ such that $T(N)\leq cf(N)$ when $N\geq n_0$
	- Consider $1000N$ and $N^2$. There are values for $N$ where $1000*N$ is larger, but $N^2$ *grows* faster
	- There is some points, $n_0$, after which $N^2$ is always larger than $1000N$
- If $T(N)=1000N$ and $f(N)=N^2$, $T(N)\leq cf(N)$ when:
	- $c=1$ and $n_0=1000$, $c=100$ and $n_0=10$
	- Explanation:
		- When $N=1000$, then $T(N)=f(N)$, after that f(N) is always larger
```run-python
import matplotlib.pyplot as plt
import numpy as np

# Define the range for N
N = np.linspace(1, 1000, 1000)  # We plot from 1 to 1000 for N

# Define the functions T(N) = 1000N and f(N) = N^2
T_N = 1000 * N
f_N = N**2

# Plot T(N)
plt.plot(N, T_N, label='T(N) = 1000N')

# Plot f(N) with c=1 and c=100 for comparison
plt.plot(N, f_N, label='f(N) = N^2 (c=1)')
plt.plot(N, 100 * f_N, label='cf(N) = 100N^2 (c=100)')

# Add a legend, titles, and labels
plt.legend()
plt.xlabel('N')
plt.ylabel('Value')
plt.title('Comparison of T(N) and f(N) with different constants c')
plt.grid(True)
plt.show()

```
- $T(N)=(Ωg(N))$ if there are positive constants $c$ and $n_0$ such that $T(N) \geq cg(N)$ when $N \geq n_0$
	- **Big Omega (Ω)**: When we say "$T(N) = Ω(g(N))$", we're saying that for large enough N (specifically, when N is larger than some starting point $n_0$), the function $T(N)$ will be at least as big as a constant times $g(N)$. The constant here is denoted by "$c$". This is like saying "$T(N)$ won't grow slower than $g(N)$ times some constant factor."
- $T(N)=(Θh(N))$ if and only if
	- $T(N)=O(h(N))$ and
	- $T(N)=Ω(H(N))$
		- **Big Theta (Θ)**: The statement "$T(N) = Θ(h(N))$" means that $T(N)$ grows at the same rate as $h(N)$. For $T(N)$ to equal $Θ(h(N))$, it must be both $O(h(N))$ and $Ω(h(N))$. This means $T(N)$ is sandwiched between two constant multiples of $h(N)$ — it doesn't grow significantly faster or slower than $h(N)$.
			- "$T(N) = O(h(N))$" means that $T(N)$ doesn't grow faster than some constant times $h(N)$.
			- "$T(N) = Ω(h(N))$" means that $T(N)$ doesn't grow slower than some constant times $h(N)$.
		- When both these conditions are met, we can confidently say that $T(N)$ grows at the same rate as $h(N)$, which is what Big Theta means.
	- In simpler terms, think of $T(N)$ as the time it takes to bake $N$ cakes. Big Omega would mean that no matter how many cakes you bake, it won't take less time than some constant times $g(N)$. Big Theta is like having a perfect recipe that always takes the same time per cake, give or take a constant multiple, whether you're baking a small or a large batch.
### Tighter Bounds
- If $f(N)=2N$, then technically
	- $F(N)=O(N^2)$,
	- $F(N)=O(N^3)$ and so on
- $f(N)=O(N)$ is the best option
### Conventions
- Do not include lower order terms and constants
	- If $f(N)=2N$, write $f(N)=O(N)$
	- If $f(N)=N^3+N+7$, write $f(N)=O(N^3)$
### Bounds
- Based off the graph we can see that red is an upper bound since it is always larger so it is O for the twosum_si while the green line is always lower making it Ω.
	- Since twosum_si is $O(N^2)$ and $Ω(N^2)$ it is also $Θ(N^2)$
		- This only applies to this specific implementation of two sum.
![[Pasted image 20240226215907.png]]
### A grain of salt
- We sometimes can find a tighter bound if analysis is an overestimation
- In some cases average case analysis if very complex
- The worst case bound is the bets analytical result known
## Lecture 3 (Lists, Stacks, and Queues) Ch. 3
### Abstract Data Type (ADT)
- An abstract data type defines a class of abstract objects which is completely characterized by the operations available on those objects. This means that an abstract data type can be defined by defining the characterizing operations for that type. […] Implementation information, such as how the object is represented in storage, is only needed when defining how the characterizing operations are to be implemented. The user of the object is not required to know or supply this information.
- An ADT is a theoretical (mathematical) model
- It defines the behavior of a data type, not its implementation
	- The type is completely opaque from outside of its operations
- The behavior is defined from the user's perspective, i.e., as a set of operations.

- A data type consists of a set of values or elements/the domain
	- If finite, this can be specified via enumeration
	- If not, via some rule describing the elements
- and a set of operations
	- Syntax, names and values
	- Semantics, functionality/behavior
- Example
	- Integer can be considered an abstract data type
		- The domain is $ℤ$ (the integers)
	- The operations include
		- +,-,*,/
		- =,!=,<,>
#### Semantics
- Semantics can be specified in a few different ways
	- Text, Algebraic specification, or Operational specification
- Example: Algebraic specification
	- ![[Pasted image 20240312115249.png]]
#### Abstract vs Concrete (DT)
- An ADT is a model, not an implementation
- It is a theoretical tool, not a language construct
- The model can be implemented in multiple ways
	- Sometimes with restrictions
		- This can be such as numpy vector implementation will overflow for large enough numbers
### Lists
- A List is a finite, ordered sequence of elements (ordered meaning each element has a position not that its sorted necessarily)
#### Simple List (Array/memory based)
- We can create a simple list by simulating memory that we can use to store the list
```python
class SimpleList():
	def __init__(self) -> None:
		self.lst = np.empty(10, dtype=np.int64)
		self.cnt = 0
	def append(self:SimpleList, d:int) -> None:
		self.lst[self.cnt] = d
		self.cnt += 1
```
#### Free List
- When we delete from this simple list we would have to move everything after the deletion left so avoid holes.
- If we want to avoid this we can try to hold a free list which is just a list that contains true or false for each position to say if its free. But now to append we must iterate over the list to find a free spot.
#### Increase Length
- When we go out of the current size we can increase the size. Typically doing this by 1 is bad since we need to copy over the original list every time. So, maybe double the list length when the size limit is reached.
#### Simple Single Linked List
```run-python
from dataclasses import dataclass

@dataclass
class LLNode:
    val: int
    nxt: 'LLNode | None' = None

class LinkedList:
    def __init__(self) -> None:
        self.lst = None
        
    def append(self, d: int) -> None:
	    # O(n)
        if self.lst is None:
            self.lst = LLNode(d)
        else:
            p = self.lst
            while p.nxt is not None:
                p = p.nxt
            p.nxt = LLNode(d)

    def delete(self, d: int) -> None:
	    # O(n)
        if self.lst is not None:
            if self.lst.val == d:
                if self.lst.nxt is None:
                    self.lst = None
                else:
                    self.lst = self.lst.nxt
            else:
                p = self.lst
                while p.nxt is not None and p.nxt.val != d:
                    p = p.nxt
                if p.nxt is not None:
                    p.nxt = p.nxt.nxt
    def print_list(self) -> None:
	    if self.lst is None:
	        print("LinkedList is empty")
	    else:
	        p = self.lst
	        while p is not None:
	            print(p.val, end=" -> ")
	            p = p.nxt
	        print("None")


ll = LinkedList()
ll.print_list()
ll.append(1)
ll.append(2)
ll.print_list()
ll.delete(1)
ll.print_list()
ll.delete(2)
ll.print_list()
```
- We run into performance problems since we traverse the list to append or delete items.
- To try and solve this we can keep track of the last item.
#### Linked List With Last Tracker
```run-python
from dataclasses import dataclass

@dataclass
class LLNode:
    val: int
    nxt: 'LLNode | None' = None

class LinkedList:
    def __init__(self) -> None:
        self.lst = None
        self.last = None
        
    def append(self, d: int) -> None:
        # O(1)
        if self.lst is None:
            self.lst = LLNode(d)
            self.last = self.lst
        else:
            self.last.nxt = LLNode(d)
            self.last = self.last.nxt

    def delete(self, d: int) -> None:
        # O(n)
        if self.lst is not None:
            if self.lst.val == d:
                if self.lst.nxt is None:
                    self.lst = None
                    self.last = None
                else:
                    self.lst = self.lst.nxt
            else:
                p = self.lst
                while p.nxt is not None and p.nxt.val != d:
                    p = p.nxt
                if p.nxt is not None:
                    p.nxt = p.nxt.nxt
                    if p.nxt is None:
                        self.last = p

    def print_list(self) -> None:
        if self.lst is None:
            print("LinkedList is empty")
        else:
            p = self.lst
            while p is not None:
                print(p.val, end=" -> ")
                p = p.nxt
            print("None")


ll = LinkedList()
ll.print_list()
ll.append(1)
ll.append(2)
ll.print_list()
```
### Stacks
- A stack is an ADT where values are inserted and removed from the "top"
- LIFO (Last In, First Out)
- Two main operations push and pop (insert and remove)
- Useful for handling function calls in programs
#### Stack using linked list
- The main operations are defined as such...
```python
__init__ (newstack)
push(v:int) -> None
pop() -> None
top() -> int|None
empty() -> bool
```

```run-python
from dataclasses import dataclass

@dataclass
class LLNode:
    val: int
    nxt: 'LLNode | None' = None

class LLStack:
    def __init__(self) -> None:
        self.stack = None
    
    def empty(self) -> bool:
        return self.stack is None
    
    def top(self) -> 'int | None':
        if self.stack is None:
            return None
        else:
            return self.stack.val
    
    def push(self, v: int) -> None:
        tmp = LLNode(v)
        tmp.nxt = self.stack
        self.stack = tmp
    
    def pop(self) -> None:
        if self.stack is not None:
            self.stack = self.stack.nxt

stack = LLStack()
assert stack.empty() == True

stack.push(1)
assert stack.empty() == False
assert stack.top() == 1

stack.pop()
assert stack.empty() == True

print("All tests pass")
```

### Queues
- A queue is an ADT where values are inserted and removed from each end. (Like a waiting line in the real world)
- FIFO (First In, First Out)
- Two main operations enqueue and dequeue (insert and remove)
- Useful for:
	- multiple requests, network traffic, processes in an OS.
#### Queue using array
- This suffers from the same issue as a simple list where when it is full it must increase size somehow and copy over the old contents.
```run-python
import numpy as np

class SimpleQueue:
    def __init__(self) -> None:
        self.size = 10
        self.lst = np.empty(self.size, dtype=np.int64)
        self.front = 0
        self.back = 0

    def enqueue(self, v: int) -> None:
        if self.back < self.size:
            self.lst[self.back] = v
            self.back += 1

    def dequeue(self) -> 'int | None':
        if self.front < self.back:
            tmp = self.lst[self.front]
            self.front += 1
            return tmp
        return None

    def count(self) -> int:
        return self.back - self.front

    def empty(self) -> bool:
        return self.count() == 0

q = SimpleQueue()
assert q.empty() == True

q.enqueue(1)
assert q.empty() == False
assert q.count() == 1

assert q.dequeue() == 1
assert q.empty() == True

print("All tests pass")

# Test of how bad it is
q = SimpleQueue()
q.enqueue(0)
for i in range(1, 15):
	q.enqueue(i)
	q.dequeue()

print(q.count())
```
- We run into an issue where we can run into unused space when we use dequeue in this simple implementation.
- If we increase the size of the backing array we waist a lot of space
#### Circular Queue
- The back and front move as we en/dequeue
- A lot of space could be wasted, so instead we reclaim the space by wrapping around when we hit the end.
- Example
	- Queue of size 10, back at 9 front and 7
	- Do not increase array size
	- Instead, insert at (9+1) % 10
		- This inserts back at index 0
```run-python
import numpy as np

class SimpleQueue:
    def __init__(self) -> None:
        self.size = 10
        self.lst = np.empty(self.size, dtype=np.int64)
        self.front = 0
        self.back = 0

    def enqueue(self, v: int) -> None:
        if self.count() < self.size:
            self.lst[self.back] = v
            self.back = (self.back + 1) % self.size

    def dequeue(self) -> 'int | None':
        if self.count() > 0:
            tmp = self.lst[self.front]
            self.front = (self.front + 1) % self.size
            return tmp
        return None

    def count(self) -> int:
	    if self.back >= self.front:
	        return self.back - self.front
	    else:
		    return self.size - (self.front - self.back)

    def empty(self) -> bool:
        return self.count() == 0

q = SimpleQueue()
assert q.empty() == True

q.enqueue(1)
assert q.empty() == False
assert q.count() == 1

assert q.dequeue() == 1
assert q.empty() == True

q = SimpleQueue()
q.enqueue(0)

gots = []
for i in range(1, 1001):
	q.enqueue(i)
	gots.append(q.dequeue())

assert gots == list(range(1000))

print("All tests pass")
```
- This can fill up though, so enqueue should extend when full and copy over the data
#### Linked Queue
```run-python
from dataclasses import dataclass

@dataclass
class LLNode:
    val: int
    nxt: 'LLNode | None' = None

class LinkedQueue:
    def __init__(self) -> None:
        self.front = None
        self.back = None
        self.cnt = 0

    def count(self) -> int:
        return self.cnt

    def enqueue(self, v: int) -> None:
        if self.back is None:
            self.back = LLNode(v)
            self.front = self.back
        else:
            self.back.nxt = LLNode(v)
            self.back = self.back.nxt
        self.cnt += 1

    def dequeue(self) -> 'int | None':
        if self.front is not None:
            tmp = self.front.val
            self.front = self.front.nxt
            if self.front is None:
                self.back = None
            self.cnt -= 1
            return tmp
        return None

# Example usage:
q = LinkedQueue()
q.enqueue(1)
q.enqueue(2)
print(f"Front value after enqueues: {q.front.val}")  # Should be 1

dequeue_val = q.dequeue()
print(f"Dequeued value: {dequeue_val}")  # Should be 1
print(f"New front value: {q.front.val}")  # Should be 2 if not None
print(f"Queue count: {q.count()}")  # Should be 1

```

### Set/Bag
#### Set
- A set is an unordered collection of unique elements
- The main operation is set membership
	- as well as way to combine sets, e.g., union, intersection, and difference
- Mutable and immutable
```run-python
class LLNode:
    def __init__(self, val, nxt=None):
        self.val = val
        self.nxt = nxt

class LLIterator:
    def __init__(self, node):
        self.current = node

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is None:
            raise StopIteration
        else:
            val = self.current.val
            self.current = self.current.nxt
            return val

class LLSet:
    def __init__(self):
        self.lst = None

    def add(self, v):
        if self.lst is None:
            self.lst = LLNode(v)
        else:
            p, pp = self.lst, None
            found = False
            while p is not None:
                if p.val == v:
                    found = True
                    break
                pp = p
                p = p.nxt
            if not found:
                new_node = LLNode(v)
                if pp is not None:  # pp is None when list is empty
                    pp.nxt = new_node
                else:
                    self.lst = new_node

    def delete(self, v):
        if self.lst is not None:
            if self.lst.val == v:
                self.lst = self.lst.nxt
            else:
                p, pp = self.lst, None
                while p is not None:
                    if p.val == v:
                        if pp is not None:  # If pp is None, it means p is the first node
                            pp.nxt = p.nxt
                        return
                    pp, p = p, p.nxt

    def contains(self, v):
        p = self.lst
        while p is not None:
            if p.val == v:
                return True
            p = p.nxt
        return False
    
    def __contains__(self, v):
        return self.contains(v)
        
    def __iter__(self):
        return LLIterator(self.lst)

# Example usage and iteration over the set
s = LLSet()
s.add(1)
s.add(2)
s.add(3)

for v in s:
    print(v)
```
#### Bags
- Checking for duplicates is expensive so a bag is a set that allows for duplicates
- Insert easier but some other operations more complex like delete which must find all instances
- If lots of adds expected, this can still save time but at the cost of space
## Lecture 4 (Trees) Ch. 4.1-4.6
- Useful for File/Directory structure
- HTML/DOM
- Parse tree
### Tree ADT
- A tree is a collection of nodes
- If not empty then,
	- it has a root node `r`
	- and zero or more sub trees that are connected from the root by a directed edge
- The root of each sub tree is a child of `r`, and `r` is the parent of each sub tree
- Each sub tree is a tree
- Node degree means the number of children
- ![[Pasted image 20240318192358.png]]
- A node can have an arbitrary number of children
- Nodes with no children are called leaves
- Nodes with the same parent are siblings
### Paths
- A path from node $n_1$ to node $n_k$ is defined as a sequence of nodes:
	- $n_1,n_2,...,n_k$
	- $n_i$ is the parent of $n_{i+1}$ for $i<=i<k$
- The length of a path is the number of edges it contains
	- So, the length of $n_1,...,n_k$ is $k-1$
- The depth of a node, is the length of the path from the root to the node
- The height of a node is the longest path from the node to a leaf
	- All leaves have height of 0
	- The height of the tree is the height of the root
- If there is a path from $n_i$ to $n_j$ then
	- $n_i$ is an ancestor of $n_j$
	- $n_j$ is an descendant of $n_i$
- If $n_i $\ne$ n_j$ then they are proper, e.g., proper ancestor
### LCRS Tree
- Not good to keep references to all children in the node so we can do left most child, right sibling (also known as first child, next sibling)
- Keep two pointers in each node
	- left child
	- right sibling
- ![[Pasted image 20240318193402.png]]
- ![[Pasted image 20240318193415.png]]
#### Implementation
```python
from dataclasses  import dataclass
@dataclass
class LCRSNode:
	key: int
	left: 'LCRSNode|None' = None
	right: 'LCRSNode|None' = None
```
##### Walking the tree
```python
from fastcore.basics import patch
@patch
def walk(root:LCRSNode) -> None:
	if root is not None:
		print(root.key)
		walk(root.left)
		walk(root.right)
```
##### Adding children
```python
@patch
def add_children(self, key) -> LCRSNode:
	if self.left is None:
		self.left = LCRSNode(key)
		return self.left
	else:
		p = self.left
		while p.right is not None:
			p = p.right
		p.right = LCRSNode(key)
		return p.right
```
##### Is a node a leaf?
```python
@patch
def is_leaf(self) -> bool:
	return self.left is None
```
##### Size and Height
```python
@patch
def size(self) -> int:
	l, r = 0, 0
	if self.left is not None:
		l = self.right.size
	if self.right is not None:
		r = self.right.size
	return l + r + 1

def height(self) -> int:
	h, p = 0, self.left
	p = self.left
	while p is not None:
		h = max(h, 1 + p.height)
		p = p.right
	return h
```
### Binary Search Trees
- This is a tree where each node has at most two children
- We can reason about height:
	- an average binary tree has height $\theta (\sqrt n)$
	- a "full" tree has height $\left\lceil \log_2(n) \right\rceil - 1$
		- That notation means ceiling
		- ![[Pasted image 20240318195235.png]]
	- a "degenerate" tree has height $n-1$
		- Essentially just a linked list
		- ![[Pasted image 20240318195258.png]]
#### Implementation

> [!NOTE]
> This first part is just a binary tree NOT a binary search tree.

```python
@dataclass
class BTNode:
	key: int
	left: 'BTNode|None' = None
	right: 'BTNode|None' = None
```
##### In order traversal
```run-python
from dataclasses import dataclass

@dataclass
class BTNode:
    key: int
    left: 'BTNode | None' = None
    right: 'BTNode | None' = None

def inorder(r: BTNode):
    if r is None:
        return ''
    else:
        s = inorder(r.left)
        s += f' {r.key} '
        s += inorder(r.right)
        return s

def preorder(r: BTNode):
    if r is None:
        return ''
    else:
        s = f' {r.key} '
        s += preorder(r.left)
        s += preorder(r.right)
        return s

def postorder(r: BTNode):
    if r is None:
        return ''
    else:
        s = postorder(r.left)
        s += postorder(r.right)
        s += f' {r.key} '
        return s

# Constructing a binary tree:
#        1
#       / \
#      2   3
#     / \
#    4   5

root = BTNode(1)
root.left = BTNode(2)
root.right = BTNode(3)
root.left.left = BTNode(4)
root.left.right = BTNode(5)

# Testing the tree traversal functions
print("Inorder traversal: ", inorder(root).strip())
print("Preorder traversal: ", preorder(root).strip())
print("Postorder traversal: ", postorder(root).strip())

```
#### Binary Search Tree
##### How do we insert?
- Put smaller to the left and larger to the right. This is a BST
- ![[Pasted image 20240318200943.png]]
```python
# Note: This wont insert mutliple of the samevalue
def _add(self, n, key):
	if n is None:
		return BTNode(key)

	if n .key > key:
		n.left = self._add(n.left, key)
	elif n.key < key:
		n.right = self._add(n.right, key)

	return n
```
##### Check if value exists
```python
def _contains_(self, key):
	return self._contains(self.root, key)
```
##### Delete
- Assume we want to delete 6
- If the node has one child, we "lift" it
	- ![[Pasted image 20240318202029.png]]
- This is harder to do when it has two children
	- We replace the node with the smallest value in the right sub tree
	- ![[Pasted image 20240318202122.png]]
##### Find the smallest node in a sub tree
```python
def _min(self, n):
	if n.left is None:
		return n.key
	else:
		return self._min(n.left)
```
##### Recursive Delete
```python
def _delete(self, n, key):
	if n in None:
		return None
	if n.key > key:
		n.left = self._delete(n.left, key)
	elif n.key < key:
		n.right = self._delete(n.right, key)
	else:
		if n.right is None:
			return n.left
		if n.left is None:
			return n.right
		n.key = self._min(n.right)
		n.right = self.delete(n.right, n.key)
	return n
```
##### Height
```python
def _height(self, n):
	if n is None:
		return -1
	else:
		return 1 + max(self._height(n.left), self._height(n.right))
```
##### A Balanced Tree?
- If we are just adding nodes to the tree we can see that we are much closer to the best case rather than the worst case for height.
	- ![[Pasted image 20240318202731.png]]
- When we add in delete we can see that the height gets worse.
	- ![[Pasted image 20240318202800.png]]
##### Operations
- The cost of all operations depends on the height of the tree
- For balanced trees, all operations are $O(log(n)$
- For degenerate trees, all operations are $O(n)$
- We know that average trees are rarely balanced or degenerate
- If we allow deletes, an average tree has height $O(\sqrt(n)$j
### AVL-Trees
- Adelson-Velskii and Landis
- This is a binary search tree with a balance condition
#### Balance Condition
- This ensures that the depth of the tree is $O(logN)$
- Must be easy to maintain
- First idea, the left and right sub trees should be the same height
	- This can result in poorly balanced trees
	- ![[Pasted image 20240318203427.png]]
- Balance at the root is not enough so each node should have left and right sub trees of the same height + or - 1
#### AVL-Trees
- The height of the left and right sub trees can differ at most 1
- Gives a height of about $1.44*log_2(N+2)-1.328$
	- This is more than $log_2$, but not that much
- Minimum nodes at a height are:
	- $S(h)=S(h-1)+s(h-2)+1$
	- So a tree with height 9 has at least 143 nodes
- ![[Pasted image 20240318203950.png]]
##### Implementation
- Inserting can destroy balance so, insert must make sure the tree remains balanced after insert
- There are four possible cases: insert into left (L) sub tree of left (L) child, LR, RL, and RR
	- Two are symmetric: LL and RR, and LR and RL
	- And one pair is easier, LL and RR
###### Single rotation (LL and RR)
- ![[Pasted image 20240318204236.png]]
- What is going on?
	- Node $k_2$ (the root) is violating the balance condition
		- since X is two levels deeper than Z
		- A change to X caused the violation
	- We can fix this by moving X higher and Y and Z lower
	- Single rotation
		- This means $k_1$ becomes root
		- and $k_2$ its right child, since $k_1<k_2$
		- Y becomes the left child of $k_2$ since $k_1<Y<k_2$
		- ![[Pasted image 20240318204956.png]]
		- ![[Pasted image 20240318205138.png]]
	- Double rotation
		- Previously, we have seen LL and RR
		- For RL and LR we need to rotate twice
			- for RL, "double right"
			- Double right means
				- rotate right child left
				- rotate self right
			- ![[Pasted image 20240318205542.png]]
##### AVL Implementation
```python
@dataclass
class AVLNode:
	key: int
	left: 'AVLNode|None' = None
	right: 'AVLNode|None' = None
	height: int = 0

class AVLTree:
	def __init__(self):
		self.root = None

	def _height(self, n):
		if n is None:
			return -1
		return n.height

	def _add(self, n, key):
		if n is None:
			return AVLNode(key)

		if n.key > key:
			n.left = self._add(n.left, key)
		elif n.key < key:
			n.right = self._add(n.right, key)
		return self._balance(n)

	def _delete(self, n, key):
		if n is None:
			return None

		if n.key > key:
			n.left = self._delete(n.left, key)
		elif n.key < key:
			n.right = self._delete(n.right, key)
		else:
			if n.right is None:
				return n.left
			if n.left is None:
				return n.right
			n.key = self._min(n.right)
			n.right = self._delete(n.right, n.key)

		return self._balance(n)

	def _rotate_left(self, r2):
		r1 = r2.left
		r2.left = r1.right
		r1.right = r2
		r2.height = max(self._height(r2.left), self._height(r2.right)) + 1
		r1.height = max(self._height(r1.left), r2.height) + 1

		return r1

	def _rotate_right(self, r2):
		r1 = r2.right
		r2.right = r1.left
		r1.left = r2
		r2.height = max(self._height(r2.left), self._height(r2.right)) + 1
		r1.height = max(self._height(r1.left), r2.height) + 1

		return r1

	def _double_right(self, n):
		n.right = self._rotate_left(n.right)
		return self._rotate_right(n)

	def _double_left(self, n):
		n.left = self._rotate_right(n.left)
		return self._rotate_left(n)

	def _balance(self, n):
		if n is None:
			return n

		if self._height(n.left) - self._height(n.right) > 1:
			if self._height(n.left.left) >= self._height(n.left.right):
				n = self._rotate_left(n)
			else:
				n = self._double_left(n)
		elif self._height(n.right) - self._height(n.left) > 1:
			if self._height(n.right.right) >= self._height(n.right.left):
				n = self._rotate_right(n)
			else:
		AVL		n = self._double_right(n)

		n.height = max(self._height(n.left), self._height(n.right)) + 1
		return n
```
- With this a tree with 1023 nodes should have a height of about 13
	- $1.44*log_21023-1.1328$
- Running a 100,000 inserts, we find that the height is between 10 to 12
	- mean is 11.000310, so very close to 11
	- Compared to a mean of 33.5 with binary search trees and no balancing effort
- The above is **with** deletes
- ![[Pasted image 20240318211856.png]]
### Splay-Trees
- Many applications have data locality
	- A node is accessed multiple times withing a reasonable time frame
- Splay trees push a node to the root after it is accessed
- Uses a series of rotations from AVL trees
- Can also help balance the tree
- Amortized Cost
	- Splay trees guarantees that m consecutive operations is $O(,log_2n)$
	- A single operation can still be $\theta(n)$, so the bound is not $O(log_2n)$
	- This is called amortized running time
		- if m operations are $O(m*f(n))$
		- the amortized cost is $O(f(n))$
- If an operation is $O(n)$ and we want $O(log_2n)$ it is clear that we must do something to fix it
	- In splay trees, we fix by moving
	- So, if first $O(n)$, then consecutive close to $O(1)$
	- ![[Pasted image 20240318212603.png]]
	- ![[Pasted image 20240318212612.png]]
	- We can move any node to the root by combining zig, zig-zig, and zig-zag
	- We do this each time we search for a node
	- This will ensure that nodes that we have searched for will be closer to the root and be quicker to find again.