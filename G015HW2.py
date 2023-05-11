import random

from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import statistics
import time


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
    a = rand.randint(1, p-1)
    b = rand.randint(0, p-1)

    def hash(u):
        return ((((a * u) + b) % p) % C)
    
    def edge_same_color(hash: callable, edge):
        colors = hash(edge[0]), hash(edge[1])
        if colors[0] == colors[1]:
            return [(colors[0], edge)]
        else:
            return []
    
    triangle_count = (edges.flatMap(lambda edge: edge_same_color(hash, edge))  # <-- MAP PHASE (R1)
                    .groupByKey()  # <-- GROUPING
                    .map(lambda edges: (0, CountTriangles(edges[1])))  # <-- REDUCE PHASE (R1)
                    .reduceByKey(lambda x,y: x + y))  # <-- REDUCE PHASE (R2)

    return triangle_count.collect()[0][1] * C ** 2

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
    rawData = sc.textFile(data_path)  
    # transform it into an RDD of edges
    edges = rawData.map(lambda line: tuple(map(int, line.split(",")))).sortBy(lambda _: random.random()) #shuffle the data
    edges = edges.repartition(numPartitions=C).cache() 


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
        node_color_estimates.append(MR_ApproxTCwithNodeColors(edges, C))
        finish_time = time.time()
        times += (finish_time - start_time)
    print(f'- Number of triangles (median over {R} runs) = ', statistics.median(node_color_estimates)) # get the median
    print(f'- Running time (average over {R} runs) = ', int((times / R) * 1000), "ms") # get the average time


if __name__ == "__main__":
    main()
