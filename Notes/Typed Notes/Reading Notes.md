# Chapter 8 (The Disjoint Set Class)
## 8.1 Equivalence Relations
- **Efficient Data Structure for Equivalence Problem**: Describes a simple, yet fast data structure for solving equivalence, notable for its easy implementation and efficient operation, utilizing just a simple array for constant average time per operation.

- **Running Time Analysis**: The analysis of the running time for this fast implementation is challenging, indicating the data structure's unique theoretical interest due to the unusual form of its worst-case scenario.

- **Equivalence Relations Explained**: An equivalence relation is defined as a relation that satisfies three specific properties:
  - **Reflexive**: Every element is related to itself.
  - **Symmetric**: If one element is related to another, then the second is related to the first.
  - **Transitive**: If an element is related to a second, which in turn is related to a third, then the first element is related to the third.

- **Examples of Equivalence Relations**:
  - **Electrical Connectivity**: Described as an equivalence relation because if one component is electrically connected to another, the connection is mutual, and if a chain of connections exists, the start and end components are connected.
  - **Geographic Location**: Cities or towns within the same country are considered equivalent, showcasing another instance of an equivalence relation.
  - **Road Travel**: The relation between towns connected by two-way roads is an equivalence relation due to the ability to travel mutually between them.
## 8.2 The Dynamic Equivalence Problem
- The equivalence class of an element a ∈ S is the subset of S that contains all the elements that are related to a.
- **Dynamic Equivalence Problem**: Focuses on deciding if two elements are equivalent under a given relation, challenging due to usually implicit relation definitions.
- **Equivalence Relation**: Defined by reflexivity, symmetry, and transitivity, with examples provided to illustrate both equivalence and non-equivalence relations.
- **Equivalence Classes**: Describes how each element belongs to a subset containing all elements it's related to, forming a partition of the set, which simplifies checking for equivalence.
- **Disjoint Set Strategy**: Initially, each element forms its own set, representing no relations. The strategy involves two operations: `find` (determines the set containing an element) and `union` (merges sets containing two elements into one, maintaining disjointness).
- **Union-Find Algorithm**: Dynamic as sets change over time, must operate ==online==, providing immediate responses for `find` operations. Differentiates from ==offline== algorithms that process all operations before providing responses.
	- Example: Written test is offline, Speaking test is online (you must answer the current question before moving one)
- **Efficiency Strategies**: Discusses maintaining equivalence class names in an array for fast `find` operations, and optimizing `union` operations by linking smaller equivalence classes to larger ones, aiming for efficient overall performance.
- **Application and Performance**: Mentions the importance in graph theory and compilers, with a focus on achieving efficient `find` or `union` operations but not both simultaneously in constant worst-case time.
- There are two strategies to solve this problem. One ensures that the find instruction can be executed in constant worst-case time, and the other ensures that the union instruction can be executed in constant worst-case time. It has been shown that both cannot be done simultaneously in constant worst-case time.

> [!Disjoint Set Time Complexity]
> - ***Needs clarification:***
> 	- For quick find a sequence of N − 1 unions (the maximum, since then everything is in one set) would take Θ($N^2$) time.
> 	- If we also keep track of the size of each equivalence class, and when performing unions we change the name of the smaller equivalence class to the larger, then the total time spent for N − 1 merges is O(N log N).
> 	- Organizing elements into linked lists based on their equivalence class. This method improves update times by avoiding full array searches but doesn't inherently reduce asymptotic running time. The breakthrough comes with tracking the size of each equivalence class and preferring to merge smaller classes into larger ones during union operations. This strategy ensures that an element's class can change at most log⁡ N times, leading to a total time of O(N log⁡ N) for N−1 merges. The approach balances the cost of find and union operations, aiming for an overall running time slightly over O(M+N) for M finds and N−1 unions.
## 8.3 Basic Data Structure
- The problem requires that `find` operations on any two elements return the same result if they are in the same set, without needing to return a specific name.
- Utilizes trees to represent sets, where all elements in a tree share the same root, simplifying set identification to checking root names.
- Initially, each set is a single-element tree, forming a forest with implicit representation in an array where each element points to its parent, and root elements point to themselves with a special marker (e.g., -1).
- Union operations merge two sets by linking the roots of their trees, aiming for constant time execution by adjusting parent pointers.
- The `find` operation's time complexity depends on the tree's depth, with potential for deep trees making `find` slow in the worst case.
- Discusses strategies to ensure efficient union operations without causing excessive `find` operation slowdowns, considering various models for analyzing average-case scenarios and their implications on the likelihood of merging larger trees.
## 8.4 Smart Union Algorithms
- **Union-By-Size Improvement**: Prioritizes merging smaller trees into larger ones to maintain shallower tree depths, enhancing efficiency.
- **Depth Limitation**: Ensures the maximum depth of any node does not exceed log N, enhancing find operation efficiency to O(log N).
- **Implementation Details**: Utilizes an array to track tree sizes, storing the negative size at each root, simplifying union operations without additional space requirements.
- **Union-By-Height Alternative**: Similar to union-by-size but tracks tree height instead, ensuring all trees have a maximum depth of O(log N) with straightforward implementation.
## 8.5 Path Compression
- **Path Compression Introduction**: Highlights path compression as an enhancement for the find operation in the union/find algorithm, independent of union strategies.
- **Operational Mechanism**: During find(x), path compression changes every node on the path from x to the root to directly point to the root, optimizing future accesses.
- **Compatibility with Union-By-Size**: Path compression works well with union-by-size, potentially speeding up operations by reducing tree depth and improving access times.
- **Union-By-Height and Path Compression**: Notes a compatibility issue with union-by-height due to path compression altering tree heights, leading to a preference for union-by-rank as an efficient alternative.
## 8.7 An Application
- **Maze Generation Example**: Illustrates maze creation using the union/find structure, where cells are initially separated by walls.
- **Algorithm Overview**: Begins with a fully walled maze except for entrance/exit, then randomly removes walls between unconnected cells until all cells are interconnected, ensuring one large connected maze.
- **Union/Find Application**: Utilizes union/find to track connected cells, ensuring walls are only removed to connect disjoint sets, preventing unnecessary wall removal between already connected cells.
- **Process Visualization**: Demonstrates through a 5x5 maze example, initially placing each cell in its own set, then progressively merging sets as walls are removed to connect cells.
- **Algorithm Conclusion**: Completes when all cells are reachable from any other, marked by all cells being in a single set, generating a complex maze with multiple paths.
- **Efficiency and Running Time**: The running time is influenced by union/find operations, with the number of find operations roughly between 2N to 4N for N cells, leading to an efficient O(N log* N) running time for maze generation.
## Summary
- Exercises 355 Summary We have seen a very simple data structure to maintain disjoint sets. When the union operation is performed, it does not matter, as far as correctness is concerned, which set retains its name. A valuable lesson that should be learned here is that it can be very important to consider the alternatives when a particular step is not totally specified. The union step is flexible; by taking advantage of this, we are able to get a much more efficient algorithm. Path compression is one of the earliest forms of self-adjustment, which we have seen elsewhere (splay trees, skew heaps). Its use is extremely interesting, especially from a theoretical point of view, because it was one of the first examples of a simple algorithm with a not-so-simple worst-case analysis.
# Chapter 2 (Algorithm Analysis)
- **Definition of an Algorithm:**
	- A set of clearly defined, simple instructions to solve a problem.
	- It must be correct and the resource requirements (time, space) should be reasonable.
	- Algorithms needing excessive resources, like a year or hundreds of gigabytes of memory, are impractical.
## 2.2 (Model)
- ![[Pasted image 20240227193557.png]]
- ![[Pasted image 20240227193754.png]]
	- These are just describing the bounds for each notation. Big Oh, Omega, Theta, and little-o
	- The idea of these definitions is to establish a relative order among functions. Given two functions, there are usually points where one function is smaller than the other function, so it does not make sense to claim, for instance, $f(N) < g(N)$. Thus, we compare their relative rates of growth.
	- When we say that T(N) = O(f(N)), we are guaranteeing that the function T(N) grows at a rate no faster than f(N); thus f(N) is an upper bound on T(N). Since this implies that $f(N) = Ω(T(N))$, we say that T(N) is a lower bound on f(N). 
	- As an example, $N^3$ grows faster than $N^2$ , so we can say that $N^2 = O(N^3 )$ $or $N^3 = Ω(N^2)$
- ![[Pasted image 20240227194423.png]]