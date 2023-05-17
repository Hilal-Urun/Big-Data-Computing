from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import statistics
import time


def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    # We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)
    # Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    # Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:
        u, v = edge
        node_colors[u] = ((rand_a * u + rand_b) % p) % num_colors
        node_colors[v] = ((rand_a * v + rand_b) % p) % num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors == triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def MR_ApproxTCwithNodeColors(edges, C):
    """
    input:
    edge: a RDD of edges
    C: Number of paritions

    return:
    t_final: estimation of number of triangles

    """
    p = 8191
    a = rand.randint(1, p - 1)
    b = rand.randint(0, p - 1)

    def hash_f(u):
        return (((a * u) + b) % p) % C

    def edge_same_color(edge):
        colors = hash_f(edge[0]), hash_f(edge[1])
        if colors[0] == colors[1]:
            return [(colors[0], edge)]
        else:
            return []

    triangle_count = (edges.flatMap(lambda edge: edge_same_color(edge))  # <-- MAP PHASE (R1)
                      .groupByKey()  # <-- GROUPING
                      .map(lambda edges: (0, CountTriangles(edges[1])))  # <-- REDUCE PHASE (R1)
                      .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R2)

    return triangle_count.collect()[0][1] * C ** 2


def MR_ExactTC(edges, C):
    p = 8191
    a = rand.randint(1, p - 1)
    b = rand.randint(0, p - 1)

    def hash_f(u):
        return (((a * u) + b) % p) % C

    exact_triangle_count = (edges.flatMap(
        lambda edge: [(tuple(sorted((hash_f(edge[0]), hash_f(edge[1]), i))), edge) for i in
                      range(C)])  # <-- MAP PHASE (R1)
                            .groupByKey()  # <-- GROUPING
                            .flatMap(
        lambda x: [(x[0], countTriangles2(x[0], x[1], a, b, p, C))])  # <-- REDUCE PHASE (R1)
                            .reduce(lambda a, b: ('sum', a[1] + b[1])))  # <-- REDUCE PHASE (R2)
    return exact_triangle_count[1]


def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 5, "Usage: python G015HW2.py <C> <R> <F> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('TriangleCountExample').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)

    # 2. Read number of runs
    R = sys.argv[2]
    assert R.isdigit(), "R must be an integer"
    R = int(R)

    # 3. Read the binary flag
    F = sys.argv[3]
    assert F in ['0', '1'], "F must be either 0 or 1"
    F = int(F)

    # 3. Read input file and subdivide it into C partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path)
    # transform it into an RDD of edges
    edges = rawData.map(lambda line: tuple(map(int, line.split(","))))
    edges = edges.partitionBy(32).cache()
    # required prints in the report
    print("Dataset = ", os.path.basename(data_path))
    # SETTING GLOBAL VARIABLES
    numEdges = edges.count()
    print("Number of Edges = ", numEdges)
    print("Number of Colors = ", C)
    print("Number of Repetitions = ", R)
    if F == 0:
        # Approximation through node coloring
        print("Approximation through node coloring")
        node_color_estimates = []
        times = 0
        for i in range(R):
            start_time = time.time()
            node_color_estimates.append(MR_ApproxTCwithNodeColors(edges, C))
            finish_time = time.time()
            times += (finish_time - start_time)
        print(f"- Number of triangles (median over {R} runs) =",
              statistics.median(node_color_estimates))  # get the median
        print(f"- Running time (average over {R} runs) = ", int((times / R) * 1000), "ms")  # get the average time
        if F == 1:
        # Exact through node coloring
            print("Exact algorithm with node coloring")
        times = 0
        for i in range(R):
            start_time = time.time()
        exact_triangle_count = MR_ExactTC(edges, C)
        finish_time = time.time()
        times += (finish_time - start_time)
        print(f"- Number of triangles = ", exact_triangle_count)
        print(f"- Running time (average over {R} runs) = ", int((times / R) * 1000), "ms")  # get the average time

        if __name__ == "__main__":
            main()
