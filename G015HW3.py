import sys
import random
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

"""create_count_sketch function creates and returns a count sketch data structure as a 2D list with D rows and W 
columns """


def create_count_sketch(D, W):
    # Create a D x W count sketch data structure
    count_sketch = [[0] * W for _ in range(D)]
    return count_sketch


"""update_count_sketch function updates the count sketch data structure by incrementing the count of the given item 
using the provided hash functions. It iterates over each hash function, calculates the hash value for the item, 
and increments the corresponding cell in the count sketch. """


def update_count_sketch(count_sketch, item, hash_funcs):
    D = len(count_sketch)
    W = len(count_sketch[0])
    for i in range(D):
        hash_val = hash_funcs[i](item) % W
        count_sketch[i][hash_val] += 1


"""compute_frequencies function takes a collection of items and computes the exact frequencies of each distinct item 
in the collection and returns a dictionary where the keys are the distinct items and the values are their 
frequencies. """


def compute_frequencies(items):
    return items.countByValue()


"""compute_true_second_moment function computes the true second moment of the stream by summing the squares of the 
frequencies of all distinct items in the stream """


def compute_true_second_moment(frequencies):
    F2 = sum(freq ** 2 for freq in frequencies.values())
    return F2


"""compute_approximate_second_moment function computes the approximate second moment of the stream using the count 
sketch data structure and iterates over each cell in the count sketch, squares the count in that cell, and sums up 
all the squared counts """


def compute_approximate_second_moment(count_sketch):
    F2_tilde = sum(sum(freq ** 2 for freq in row) for row in count_sketch)
    return F2_tilde


"""compute_average_relative_error function computes the average relative error of the frequency estimates provided by 
the count sketch for items with true frequencies greater than or equal to the 𝜙(𝐾)-th largest frequency. First,
identifies the 𝜙(𝐾)-th largest frequency by sorting the frequencies in descending order and selecting the top_K 
frequencies; Then iterates over the items and true frequencies, and for each item with a true frequency greater 
than or equal to the threshold, computes the estimated frequency using the count sketch and calculates the 
relative error; Finally, computes the average relative error by summing the relative errors and dividing by the 
number of items."""


def compute_average_relative_error(frequencies, count_sketch, top_K):
    top_K_freqs = sorted(frequencies.values(), reverse=True)[:top_K]
    threshold = top_K_freqs[-1]
    relative_errors = []
    for item, true_freq in frequencies.items():
        if true_freq >= threshold:
            hash_vals = [hash_funcs[i](item) % W for i in range(D)]
            estimated_freq = min(count_sketch[i][hash_vals[i]] for i in range(D))
            relative_error = abs(true_freq - estimated_freq) / true_freq
            relative_errors.append(relative_error)
    average_relative_error = sum(relative_errors) / len(relative_errors)
    return average_relative_error


if __name__ == "__main__":
    assert len(sys.argv) == 7, "USAGE: python G015HW3.py <D> <W> <left> <right> <K> <portExp>"
    # Read command-line arguments
    D = int(sys.argv[1])
    W = int(sys.argv[2])
    left = int(sys.argv[3])
    right = int(sys.argv[4])
    K = int(sys.argv[5])
    portExp = int(sys.argv[6])

    # Define hash function parameters
    p = 8191
    a = 7
    b = 3
    C = D * W

    # Spark SETUP
    conf = SparkConf().setMaster("local[*]").setAppName('Spark_Streaming_Assignment')
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)

    # Create hash functions for count sketch
    p = 8191
    hash_funcs = [
        lambda item: ((random.randint(1, p - 1) * item) % p) % W
        for _ in range(D)
    ]

    # Create count sketch data structure
    count_sketch = create_count_sketch(D, W)

    # Create socket text stream to read the stream of integer items
    lines = ssc.socketTextStream("algo.dei.unipd.it", portExp)

    # Filter the stream to select items within the specified interval [left, right]
    filtered_items = lines.flatMap(lambda line: line.split(" ")).map(int).filter(lambda x: left <= x <= right)

    # Update count sketch and compute exact frequencies of distinct items
    filtered_items.foreachRDD(lambda rdd: update_count_sketch(count_sketch, rdd.collect(), hash_funcs))
    exact_frequencies = filtered_items.countByValue()

    # Compute statistics
    F2_true = compute_true_second_moment(exact_frequencies)
    F2_tilde = compute_approximate_second_moment(count_sketch)
    average_relative_error = compute_average_relative_error(exact_frequencies, count_sketch, K)

    # Print the required outputs
    print("Input Parameters:")
    print("D =", D)
    print("W =", W)
    print("left =", left)
    print("right =", right)
    print("K =", K)
    print("portExp =", portExp)
    print("")

    print("Stream Lengths:")
    print("|Σ| =", filtered_items.count())
    print("|Σ𝑅| =", filtered_items.count())
    print("")

    print("Distinct Items in Σ𝑅:")
    print("Number of distinct items =", len(exact_frequencies))
    print("")

    print("Average Relative Error:")
    print("Average relative error of top-K frequencies =", average_relative_error)
    print("")

    if K <= 20:
        print("True and Estimated Frequencies (Top-K):")
        top_K_items = sorted(exact_frequencies.items(), key=lambda x: x[1], reverse=True)[:K]
        for item, true_freq in top_K_items:
            hash_vals = [hash_funcs[i](item) % W for i in range(D)]
            estimated_freq = min(count_sketch[i][hash_vals[i]] for i in range(D))
            print("Item:", item)
            print("True Frequency:", true_freq)
            print("Estimated Frequency:", estimated_freq)
            print("")
