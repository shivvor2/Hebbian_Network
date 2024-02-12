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

stable_state_2 = np.array([[-1,-1,1,1],
                        [-1,-1,1,1],
                        [-1,-1,1,1],
                        [-1,-1,1,1]])

# stable_state_2 = np.array([[1,-1,1,-1],
#                         [-1,1,-1,1],
#                         [1,-1,1,-1],
#                         [-1,1,-1,1]])

# initializing the Hopfield network builder
builder = hn.HopNetBuilder()
builder.memory_add(stable_state)
builder.memory_add(stable_state_2)
builder.memory_show()
```

    Setting dimensions to (4, 4) of the added memory
    There are a total of 2 memories
    Memory 0
    □ ■ ■ □
    ■ □ □ ■
    ■ □ □ ■
    □ ■ ■ □
    Memory 1
    ■ ■ □ □
    ■ ■ □ □
    ■ ■ □ □
    ■ ■ □ □



```python
network = builder.build(init_state = "random")

network.show_state()
```

    □ ■ ■ ■
    ■ ■ □ ■
    ■ ■ ■ □
    ■ □ □ □


    /mnt/d/Work/Projects/Hopfield_Network/Hopfield_Network.py:99: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if weights != "random" and int(np.array(dims).prod()) != weights.shape[0]: # We assign weights between "all pairs" of vertices



```python
# letting the network converge to the stable state (will converge to 1 of 2 states)
network.update(wait = 0.5)
```

    Convergence Reached at iteration 1
    ■ □ □ ■
    □ ■ ■ □
    □ ■ ■ □
    ■ □ □ ■


The network can converge to other memory states as well

```python
# Trying again, this time with a different initial state
network.set_state("random")
network.show_state()

```

    ■ ■ ■ □
    □ □ ■ □
    ■ ■ ■ □
    □ ■ ■ ■



```python
network.update(wait = 0.5)
```

    Convergence Reached at iteration 0
    □ □ ■ ■
    □ □ ■ ■
    □ □ ■ ■
    □ □ ■ ■

