# Network Flow Example

This project provides a simple implementation and example
of fidning the maximum flow of a network.

This implements the Edmonds-Karp algorithm for Ford-Fulkerson
using breadth-first search and an adjacency matrix representation
of the network.

## Software Development Observation

### Why use a Record?

The first draft used side-effects in Java, where a method modifies
an object passed to it.
The `residualGraph` array was passed into the method as a mutable object
to be modified by the method.
It is a simple way to "return" multiple pieces of information,
in this case the `int flow` (return value) and the `int[][] graph`.

### Argument Against Side-Effects (clean code)

Most modern programming paradigms (especially Functional Programming) argue for immutability.
When a method modifies an object passed into it:

- Predictability decreases: A developer might not expect their residualGraph array
to be changed by a method that looks like a simple calculation.

- Debugging becomes harder: If the graph state is wrong, you have to hunt down which method modified it.

- Thread Safety: If multiple threads access that array, side-effects can lead to race conditions.

### Why do we treat returning the `parent` array and the `residualGraph` differently?

There is a subtle but important distinction in how these two arrays function within the logic:

- The `parent` array is "Transient":
It is a temporary piece of scratchpad memory used only for a single iteration of a loop.
It doesn't represent the "final result" of the algorithm; it's just a tool to get there.

- The `residualGraph` is "Stateful":
It represents the final state of the flow network.
It is the data the user actually wants to inspect after the algorithm finishes.

    `parent`:
Internal.
Hidden inside the utility logic.
The user of the class never sees it.

    `residualGraph`:
External.
User has to create this array just to call the method.
User manages memory for the algorithm: Poor "encapsulation".
