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
  - !\[\[Pasted image 20240221161447.png\]\] Three Main methods:
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
  - !\[\[Pasted image 20240221170018.png\]\] Three Main methods:
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