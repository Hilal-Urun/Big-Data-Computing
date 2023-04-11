import os
import sys

from pyspark import SparkConf, SparkContext
import time
from collections import defaultdict
import random


# CountTriangles function implementation
def CountTriangles(edges):
    # Create a default dict to store the neighbors of each vertex
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
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 3, "Usage: python G15HW1.py <C> <filename>"

    # SPARK SETUP
    conf = SparkConf().setAppName('TriangleCount')
    sc = SparkContext(conf=conf)
    # INPUT READING
    # 1. Read number of partitions
    C = sys.argv[1]
    assert C.isdigit(), "C must be an integer"
    C = int(C)

    # 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path)
    edges = rawData.map(lambda line: tuple(map(int, line.split(",")))).sortBy(lambda _: random.random())
    edges_rdd=edges.repartition(numPartitions=C).cache()

    # Call MR_ApproxTCwithSparkPartitions to get an estimate tfinal of the number of triangles in the input graph
    start_time = time.time()
    tfinal = MR_ApproxTCwithSparkPartitions(edges_rdd, C)
    end_time = time.time()

    print("File name: {}".format(data_path))
    print("Number of edges: {}".format(edges.count()))
    print("C: {}".format(C))

    # Print the estimate returned by MR_ApproxTCwithSparkPartitions and its running time
    print("Estimated number of triangles:", tfinal)
    print("Running time:", end_time - start_time)

    # Stop the SparkContext
    sc.stop()

if __name__ == "__main__":
    main()
