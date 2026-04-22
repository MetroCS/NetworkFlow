import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * A demonstration of the Edmonds-Karp algorithm to solve
 * the Maximum Flow problem using the Ford-Fulkerson approach.
 * 
 * This version supports both adjacency-matrix and adjacency-list inputs.
 * Internally, the adjacency matrix is converted to an
 * adjacency list to improve performance.
 * 
 * <PRE>For the adjacency-list version of Edmonds–Karp:
 * One BFS takes O(V+E) time, because each vertex is processed at most once
 * and each residual edge is examined at most once.
 * 
 * One augmentation therefore also takes O(V+E) time, since the path update
 * after BFS is only O(V), which does not change the bound.
 * 
 * The full Edmonds–Karp algorithm takes O(VE^2) time in the worst case, because
 * it may need as many as O(VE) augmenting paths, and each one requires a BFS.
 * 
 * single BFS: O(V+E)
 * single augmentation: O(V+E)
 * entire algorithm: O(VE^2)
 * 
 * If the input is given as an adjacency matrix but is converted once to an
 * adjacency list internally, there is also a one-time conversion cost of
 * O(V^2). That does not change the usual overall worst-case bound for
 * Edmonds–Karp, which is still reported as O(VE^2).
 * </PRE>
 *
 * @author Dr. Jody Paul
 * @version CS4050 - Spring 2026
 */
public class MaxFlowFlexibleExample {
    /** Hide the no-parameter constructor. */
    private MaxFlowFlexibleExample() { }

    /** A simple container to hold the results of a Max Flow calculation. */
    public static record MaxFlowResult(int maxFlow, Graph residualGraph) { }

    /** A minimal interface for the graph representation. */
    public interface Graph {
        /**
         * The number of edges in the graph.
         * @return the number of edges
         */
        int size();
        /**
         * The edges for the specified node.
         * @param node the index of the node
         * @return the edges incident on the specified node
         */
        List<Edge> getEdges(int node);
    }

    /**
     * Representation of a directed edge in a flow graph.
     */
    public static class Edge {
        int from;
        int to;
        int capacity;
        int originalCapacity;
        boolean isOriginalEdge;
        Edge reverseEdge;

        /**
         * Create a directed edge.
         *
         * @param from             Source vertex.
         * @param to               Destination vertex.
         * @param residualCapacity Residual capacity.
         * @param originalCapacity Original capacity of the edge.
         * @param isOriginalEdge   true if this edge is from the original graph.
         */
        public Edge(int from, int to, int residualCapacity, int originalCapacity, boolean isOriginalEdge) {
            this.from = from;
            this.to = to;
            this.capacity = residualCapacity;
            this.originalCapacity = originalCapacity;
            this.isOriginalEdge = isOriginalEdge;
        }
    }

    /** A simple adjacency-list implementation of Graph. */
    public static class AdjListGraph implements Graph {
        /** Storage for the adjacency list of this graph. */
        protected final List<List<Edge>> adjacencyList;

        /**
         * Create an empty adjacency-list graph.
         * @param numberOfNodes Number of vertices.
         */
        public AdjListGraph(int numberOfNodes) {
            adjacencyList = new ArrayList<>();
            for (int i = 0; i < numberOfNodes; i++) {
                adjacencyList.add(new ArrayList<>());
            }
        }

        /**
         * Add a directed edge and its reverse residual edge.
         * @param from     Source vertex.
         * @param to       Destination vertex.
         * @param capacity Capacity of the edge.
         */
        public void addEdge(int from, int to, int capacity) {
            Edge forwardEdge = new Edge(from, to, capacity, capacity, true);
            Edge reverseEdge = new Edge(to, from, 0, 0, false);

            forwardEdge.reverseEdge = reverseEdge;
            reverseEdge.reverseEdge = forwardEdge;

            adjacencyList.get(from).add(forwardEdge);
            adjacencyList.get(to).add(reverseEdge);
        }

        @Override
        public int size() {
            return adjacencyList.size();
        }

        @Override
        public List<Edge> getEdges(int node) {
            return adjacencyList.get(node);
        }
    }

    /**
     * A wrapper that accepts an adjacency matrix and internally
     * converts it to an adjacency-list representation.
     */
    public static class MatrixGraph extends AdjListGraph {
        /**
         * Create a graph from an adjacency matrix.
         * @param matrix The capacity matrix.
         */
        public MatrixGraph(int[][] matrix) {
            super(matrix.length);
            for (int u = 0; u < matrix.length; u++) {
                for (int v = 0; v < matrix.length; v++) {
                    if (matrix[u][v] > 0) {
                        addEdge(u, v, matrix[u][v]);
                    }
                }
            }
        }

        /**
         * Convert the current residual graph back into adjacency-matrix form.
         * @return Residual graph as a matrix.
         */
        public int[][] toResidualMatrix() {
            int[][] residualMatrix = new int[size()][size()];
            for (int u = 0; u < size(); u++) {
                for (Edge edge : getEdges(u)) {
                    if (edge.isOriginalEdge) {
                        residualMatrix[edge.from][edge.to] = edge.capacity;
                    }
                }
            }
            return residualMatrix;
        }
    }

    /** Helper structure for reconstructing an augmenting path. */
    public static class ParentInfo {
        int parentNode;
        Edge edgeUsed;

        /**
         * Create a ParentInfo object.
         * @param parentNode The predecessor node on the path.
         * @param edgeUsed   The edge used to reach the current node.
         */
        public ParentInfo(int parentNode, Edge edgeUsed) {
            this.parentNode = parentNode;
            this.edgeUsed = edgeUsed;
        }
    }

    /**
     * Determine if there is an augmenting path from the source to the sink
     * using Breadth-First Search (BFS).
     *
     * @param residualGraph The current residual graph.
     * @param source        Index of the source vertex.
     * @param sink          Index of the sink vertex.
     * @param parent        Array storing predecessor information.
     * @return              true if a path exists; false otherwise.
     * @throws ArrayIndexOutOfBoundsException if source or sink are outside the valid range.
     */
    static boolean bfs(Graph residualGraph, int source, int sink, ParentInfo[] parent) {
        boolean[] visited = new boolean[residualGraph.size()];
        Queue<Integer> queue = new LinkedList<>();
        queue.add(source);
        visited[source] = true;
        parent[source] = new ParentInfo(-1, null);

        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (Edge edge : residualGraph.getEdges(u)) {
                int v = edge.to;
                if (!visited[v] && edge.capacity > 0) {
                    parent[v] = new ParentInfo(u, edge);
                    if (v == sink) {
                        return true;
                    }
                    visited[v] = true;
                    queue.add(v);
                }
            }
        }

        return false;
    }

    /**
     * Calculate the maximum flow in a given graph from source to sink.
     * This is an implementation of the Edmonds-Karp algorithm.
     *
     * @param graph  The graph.
     * @param source Index of the source node.
     * @param sink   Index of the sink node.
     * @return       The maximum flow value and the final residual graph.
     */
    public static MaxFlowResult edmondsKarp(Graph graph, int source, int sink) {
        ParentInfo[] parent = new ParentInfo[graph.size()];
        int maxFlow = 0;

        while (bfs(graph, source, sink, parent)) {
            int pathFlow = Integer.MAX_VALUE;

            // 1. Find the bottleneck capacity on the path.
            for (int v = sink; v != source; v = parent[v].parentNode) {
                pathFlow = Math.min(pathFlow, parent[v].edgeUsed.capacity);
            }

            // 2. Update residual capacities and reverse edges.
            for (int v = sink; v != source; v = parent[v].parentNode) {
                Edge edge = parent[v].edgeUsed;
                edge.capacity -= pathFlow;
                edge.reverseEdge.capacity += pathFlow;
            }

            // 3. Add path flow to the total.
            maxFlow += pathFlow;
        }

        return new MaxFlowResult(maxFlow, graph);
    }

    /**
     * Convenience overload that accepts an adjacency matrix,
     * converts it internally, and then runs the algorithm.
     *
     * @param graph  The capacity matrix.
     * @param source Index of the source node.
     * @param sink   Index of the sink node.
     * @return       The maximum flow value and the final residual graph.
     */
    public static MaxFlowResult edmondsKarp(int[][] graph, int source, int sink) {
        return edmondsKarp(new MatrixGraph(graph), source, sink);
    }

    /**
     * Calculate and create display of the final flow on each original edge.
     * @param graph The final residual graph.
     * @return      A formatted string representing the flow/capacity for original edges.
     */
    public static String getFlowDetails(Graph graph) {
        StringBuilder sb = new StringBuilder();
        sb.append("\nEdge_# -> Flow / Capacity\n");
        sb.append("-----------------------\n");

        for (int u = 0; u < graph.size(); u++) {
            for (Edge edge : graph.getEdges(u)) {
                if (edge.isOriginalEdge) {
                    int flow = edge.originalCapacity - edge.capacity;
                    sb.append(String.format(
                            "Edge %d -> %d: %d / %d%n",
                            edge.from, edge.to, flow, edge.originalCapacity));
                }
            }
        }

        return sb.toString();
    }

    /**
     * Create display of graph in adjacency-matrix format.
     * @param result the outcome of a max-flow computation
     * @return a displayable string in adjacency-matrix format
     */
    public static String showAsAdjacencyMatrix(MaxFlowResult result) {
        StringBuilder sb = new StringBuilder();
        if (result.residualGraph() instanceof MatrixGraph matrixGraph) {
            int[][] residualMatrix = matrixGraph.toResidualMatrix();
            sb.append("Residual graph as a matrix:").append(System.lineSeparator());
            for (int u = 0; u < residualMatrix.length; u++) {
                for (int v = 0; v < residualMatrix.length; v++) {
                    sb.append(String.format("%3d", residualMatrix[u][v]));
                    if (v < residualMatrix[u].length - 1) sb.append(",");
                }
                sb.append(System.lineSeparator());
            }
        }
        return sb.toString();
    }
    
    /**
     * Main method to execute the Max-Flow example with a hardcoded graph.
     * @param args Command line arguments (ignored).
     */
    public static void main(String[] args) {
        int[][] matrix = {
            {0, 16, 13,  0,  0,  0},
            {0,  0, 10, 12,  0,  0},
            {0,  4,  0,  0, 14,  0},
            {0,  0,  9,  0,  0, 20},
            {0,  0,  0,  7,  0,  4},
            {0,  0,  0,  0,  0,  0}
        };

        MaxFlowResult result = edmondsKarp(matrix, 0, matrix.length - 1);

        System.out.println("The maximum possible flow is " + result.maxFlow());
        System.out.print(getFlowDetails(result.residualGraph()));
    }
}
