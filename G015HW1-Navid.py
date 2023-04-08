from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict

#global variables
CV = []
C = 0


def hash(num_vertices):
	
	"""
	input: 
	C = total number of colors
	num_vertices = total number of vetices in the graph
	output:
	CV = a list with size equal to number of vertices and each element of list shows the color 
				randomly assigned to a vertex. e.g. CV[1] = 2 means vertex 1 has color 2 
	"""
	global CV, C

	for i in range (num_vertices+1):
		p = 8191
		a = rand.randint(1, p-1)
		b = rand.randint(0, p-1)
		h_C = (((a*i) + b) % p) % C  
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
	if(CV[edges[0]] == CV[edges[1]]):
		edge_to_color.append((CV[edges[0]],edges))

	return edge_to_color

def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u,v = edge
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

def MR_ApproxTCwithNodeColors(edges):

	"""
	input:
	edge: a RDD of edges
	C: Number of paritions

	return:
	t_final: estimation of number of triangles
	
	"""
	global CV

	num_vertices = max(edges.max(lambda x:x[0])[0],edges.max(lambda x:x[1])[1]) #find total number of vetices to assign colors to them
	CV = hash(num_vertices) # assign colors to vertices
	
	triangle_count = (edges.flatMap(find_edges) # <-- MAP PHASE (R1)
		.groupByKey()					 # <-- SHUFFLE+GROUPING
		.mapValues(CountTriangles)    # <-- REDUCE PHASE (R1)
		.map(lambda x: x[1]).reduce(lambda a, b: a + b)) # <-- REDUCE PHASE (R2)
	
	return triangle_count


def main():

	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 4, "Usage: python G015HW1.py <C> <R> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('TriangleCountExample')
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read number of partitions
	global C
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
	rawData = sc.textFile(data_path) #not sure to keep minPartition or not ( ,minPartitions=C)
	# transform it into an RDD of edges
	edges = rawData.flatMap(lambda line: [(int(x.split(',')[0]), int(x.split(',')[1])) for x in line.split('/n')]) 
	edges = edges.partitionBy(C).cache()
    #rawData.repartition(numPartitions=C)

    #required prints in the report
	print("File name: ", os.path.basename(data_path))
	# SETTING GLOBAL VARIABLES
	numEdges = edges.count();
	print("Number of edges = ", numEdges)
	print ("C: ", C, "R: ",R)
	
	print("count: ", MR_ApproxTCwithNodeColors(edges))


if __name__ == "__main__":
	main()
