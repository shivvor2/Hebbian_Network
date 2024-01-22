## WIP

Saw [this video](https://www.youtube.com/watch?v=piF6D6CQxUw) and decided to build one myself.

Generally, A hopfield network (as described) is made up of a "grid" of fully connected neurons, each neuron taking the state of other neurons as input and changes it's own state accordingly.

To be more specific, given a hopfield network $G(V,E)$:
1. It is Fully connected i.e. $\forall v_i, v_j \in V, \exists e = \{v_i, v_j\}, e \in E$.
2. For vertex v_i, it is assigned a value 1 or -1.
3. We assign a weight matrix $W$ of $|E|$ ($ = |V| \cdot |V|$ when vertex is FC). $W [i,j]$ corresponds to the edgeweight of edge $e[v_i, v_j]$
4. For vertex v_i, the update rule is 
$$

f(\sum_{k: e \{v_k, v_i\} \in E}w_{ki} v_k)\\
f(x)= 
\begin{cases}
    1,& \text{if } x\geq 0\\
    -1,& \text{otherwise}
\end{cases}
$$
5. The vertexs updates asynchronously

Of course, we will flip some of those statements on it's head, which made some of the statements (and implementation of the hopfield network) wierd
