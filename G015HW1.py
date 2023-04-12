import random

from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import statistics
import time

# global variables
CV = []


def hash(num_vertices, C):
    """
    input:
    C = total number of colors
    num_vertices = total number of vetices in the graph
    output:
    CV = a list with size equal to number of vertices and each element of list shows the color
                randomly assigned to a vertex. e.g. CV[1] = 2 means vertex 1 has color 2
    """
    global CV

    for i in range(num_vertices + 1):
        p = 8191
        a = rand.randint(1, p - 1)
        b = rand.randint(0, p - 1)
        h_C = (((a * i) + b) % p) % C
        CV.append(h_C)

    return CV


def find_edges(edges):
    """
    input:

    edges: list of all edges
    CV: list of assgined colors to vetices

    return:
    edge_to_color: edges containing same color verices

    make a list for each color "c" and assign related edges to that list

    """
    edge_to_color = []
    global CV
    if (CV[edges[0]] == CV[edges[1]]):
        edge_to_color.append((CV[edges[0]], edges))

    return edge_to_color


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
    global CV

    num_vertices = max(edges.max(lambda x: x[0])[0],
                       edges.max(lambda x: x[1])[1])  # find total number of vetices to assign colors to them
    CV = hash(num_vertices, C)  # assign colors to vertices

    triangle_count = (edges.flatMap(find_edges)  # <-- MAP PHASE (R1)
                      .groupByKey()  # <-- SHUFFLE+GROUPING
                      .mapValues(CountTriangles)  # <-- REDUCE PHASE (R1)
                      .map(lambda x: C ** 2 * x[1]).reduce(lambda a, b: a + b))  # <-- REDUCE PHASE (R2)
    CV = []
    return triangle_count


def MR_ApproxTCwithSparkPartitions(rdd_edges, C):
    """
    :param edges: rdd edges
    :param C: number of partitions
    :return: an estimate tfinal
    """
    # Round 1
    # Compute the number of triangles for each partition
    triangle_counts = rdd_edges.mapPartitions(lambda partition: [CountTriangles(partition)])

    # Round 2
    # Compute the final estimate of the number of triangles
    tfinal = C * C * triangle_counts.reduce(lambda x, y: x + y)
    return tfinal


def main():
    # CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 4, "Usage: python G015HW1.py <C> <R> <file_name>"

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

    # 3. Read input file and subdivide it into C partitions
    data_path = sys.argv[3]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path)  # not sure to keep minPartition or not ( ,minPartitions=C)
    # transform it into an RDD of edges
    edges = rawData.map(lambda line: tuple(map(int, line.split(",")))).sortBy(lambda _: random.random())
    edges_rdd = edges.repartition(numPartitions=C).cache()
    # rawData.repartition(numPartitions=C)

    # required prints in the report
    print("Dataset = ", os.path.basename(data_path))
    # SETTING GLOBAL VARIABLES
    numEdges = edges.count();
    print("Number of Edges = ", numEdges)
    print("Number of Colors = ", C)
    print("Number of Repetitions = ", R)

    # Approximation through node coloring
    print("Approximation through node coloring")
    node_color_estimates = []
    times = 0
    for i in range(R):
        start_time = time.time()
        node_color_estimates.append(MR_ApproxTCwithNodeColors(edges_rdd, C))
        finish_time = time.time()
        times += (finish_time - start_time)
    print(f'Number of triangles (median over {R} runs) = ', statistics.median(node_color_estimates))
    print(f'Running time (average over {R} runs) = ', int((times / R) * 1000), "ms")

    # Approximation through Spark partitions
    print("Approximation through Spark partitions")
    start_time = time.time()
    tfinal = MR_ApproxTCwithSparkPartitions(edges_rdd, C)
    finish_time = time.time()
    print(f'Number of triangles = ', tfinal)
    print(f'Running time = ', int(((finish_time - start_time) / R) * 1000), "ms")


if __name__ == "__main__":
    main()
