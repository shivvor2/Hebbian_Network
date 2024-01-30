# Demo

This is the demonstration of a recreation of a Hopfield Network (as demonstrated in [this video](https://www.youtube.com/watch?v=piF6D6CQxUw))

## Imports


```python
import numpy as np
import Hopfield_Network as hn
import util
from util import print_array
```

## Basic use case:

We demonstrate hopfield network's ability to return to a stable state.

We specify a (binary, stable) state (matrix) of 1 and -1s, we then build a hopfield network with a randomized initial state and demonstrate the process of it returning to the stable state.

Initializing a state matrix


```python
# initializing a memory matrix
stable_state = np.array([[1,-1,-1,1],
                        [-1,1,1,-1],
                        [-1,1,1,-1],
                        [1,-1,-1,1]])
```

    □ ■ ■ □
    ■ □ □ ■
    ■ □ □ ■
    □ ■ ■ □
    □ □ ■ □
    □ □ ■ ■
    ■ □ ■ ■
    □ ■ □ □



```python
# initializing and building a Hopfield network
builder = hn.HopNetBuilder()
builder.add_memory(stable_state)
# builder.add_memory(stable_state2)
network = builder.build(init_state = "random")
```

    Setting dimensions to (4, 4) of the added memory


    /mnt/d/Work/Projects/Hopfield_Network/Hopfield_Network.py:85: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if weights != "random" and np.array(dims).prod() != weights.shape[0]: # We assign weights between "all pairs" of vertices



```python
network.show_state()
```

    □ ■ □ ■
    ■ □ □ □
    □ ■ ■ □
    ■ ■ □ □



```python
# letting the network converge to the stable state
network.update(wait = 0.5)
```

    Convergence Reached at iteration 0
    ■ □ □ ■
    □ ■ ■ □
    □ ■ ■ □
    ■ □ □ ■


# The Math

A hopfield network (as described in the video) is made up of a "grid" of fully connected neurons, each neuron taking the state of other neurons as input and changes it's own state accordingly.

To be more specific, given a hopfield network $G(V,E)$:
1. It is Fully connected i.e. $\forall v_i, v_j \in V, \exists e = \{v_i, v_j\}, e \in E$.
2. For vertex v_i, it is assigned a value 1 or -1.
3. We assign a weight matrix $W$ of $|E|$ (= $|V| \cdot |V|$ when vertex is FC). $W[i,j]$ corresponds to the edgeweight of edge $e[v_i, v_j]$
4. For vertex v_i, the update rule is
$f(\sum_{k: e \{ v_k, v_i \} \in E}w_{ki} v_k)$ where $f(x) = 1$ if $X > 0$ and $f(x) = 0$ otherwise 

