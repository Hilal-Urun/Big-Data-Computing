from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import statistics
import random as rand
import numpy as np
import threading
import sys

# After how many items should we stop?
THRESHOLD = 10000000

def hash_f(u, a, b, W):
    p = 8191
    return (((a * u) + b) % p) % W

def hash_g(u, a, b):
    p = 8191
    hashed_value = (((a * u) + b) % p) % 2
    return 1 if hashed_value == 0 else -1

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch, W, left, right):
    
    # We are working on the batch at time `time`.
    global streamLength, streamLength_filtered, histogram, Counter_table, hash_params_f, hash_params_g
    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0]>=THRESHOLD:
        return
    streamLength[0] += batch_size

    # map 1 to each element and filter according to right and left
    batch_maped_filtered = batch \
            .map(lambda a: (int(a), 1)) \
            .filter(lambda e: (e[0] >= left) and (e[0] <= right))
    
    streamLength_filtered[0] += batch_maped_filtered.count()

    items_count = batch_maped_filtered \
            .reduceByKey(lambda a,  b: a + b) \
            .collectAsMap()
    
    for element, count in items_count.items():
        #exact count
        if element not in histogram:
            histogram[element] = count
        else:
            histogram[element] += count

        # Update the counters with the batch
        for j in range(len(Counter_table)): # for D
            Counter_table[j][hash_f(element, hash_params_f[j][0], hash_params_f[j][1], W)] += hash_g(element, hash_params_g[j][0], hash_params_g[j][1]) * count #the hash function g (for sign) is defined here

    
    # Update the counters with the batch
    # for element, count in items_count.items():
    #     for j in range(Counter_table.shape[0]): # for D
    #         Counter_table[j, hash_f(element, hash_params[j][0], hash_params[j][1], W)] += (1 if hash_f(element, hash_params[j][0], hash_params[j][1], W) % 2 == 0 else -1) * count #the hash function g (for sign) is defined here

    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()
        


if __name__ == '__main__':

    assert len(sys.argv) == 7, "USAGE: python G015HW3.py <D> <W> <left> <right> <K> <portExp>"
    # Read command-line arguments
    D = int(sys.argv[1])
    W = int(sys.argv[2])
    left = int(sys.argv[3])
    right = int(sys.argv[4])
    K = int(sys.argv[5])
    portExp = int(sys.argv[6])
    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("DistinctExample")
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    # conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    
    
    # Here, with the duration you can control how large to make your batches.
    # Beware that the data generator we are using is very fast, so the suggestion
    # is to use batches of less than a second, otherwise you might exhaust the memory.
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)  # Batch duration of 1 second
    ssc.sparkContext.setLogLevel("ERROR")
    
    # TECHNICAL DETAIL:
    # The streaming spark context and our code and the tasks that are spawned all
    # work concurrently. To ensure a clean shut down we use this semaphore.
    # The main thread will first acquire the only permit available and then try
    # to acquire another one right after spinning up the streaming computation.
    # The second tentative at acquiring the semaphore will make the main thread
    # wait on the call. Then, in the `foreachRDD` call, when the stopping condition
    # is met we release the semaphore, basically giving "green light" to the main
    # thread to shut down the computation.
    # We cannot call `ssc.stop()` directly in `foreachRDD` because it might lead
    # to deadlocks.
    stopping_condition = threading.Event()
    
    
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    print("Receiving data from port =", portExp)
    
    
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    p = 8191
    streamLength = [0] # Stream length (an array to be passed by reference)
    streamLength_filtered = [0] # Stream length after cosidering left right (an array to be passed by reference)
    histogram = {} # 
    Counter_table = np.zeros((D, W)) 
    hash_params_f = [[rand.randint(1, p - 1), rand.randint(0, p - 1)] for _ in range(D)] #define D hash tables
    hash_params_g = [[rand.randint(1, p - 1), rand.randint(0, p - 1)] for _ in range(D)]
    
    

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch, W, left, right))
    
    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    # NOTE: You will see some data being processed even after the
    # shutdown command has been issued: This is because we are asking
    # to stop "gracefully", meaning that any outstanding work
    # will be done.
    ssc.stop(False, True)
    print("Streaming engine stopped")

    # COMPUTE AND PRINT FINAL STATISTICS
    print("D = ", D, " W = ", W, " [left,right] = [", left , ", ", right, "] K = ", K, " Port = ", portExp)
    print("Total number of items = ", streamLength[0])
    print("Total number of items in [", left, "," ,right, "]"," = ", streamLength_filtered[0])
    print("Number of distinct items in [", left,"," ,right, "]"," = ", len(histogram))

    # Calculate the K-th largest frequency
    sorted_frequencies = sorted(histogram.values(), reverse=True)
    kth_largest_frequency = sorted_frequencies[K - 1]

    # Calculate the average relative error
    total_relative_error = 0
    count = 0
    for element, count_true in histogram.items():
        if count_true >= kth_largest_frequency:
            count_estimated = statistics.median(Counter_table[j][hash_f(element, hash_params_f[j][0], hash_params_f[j][1], W)] * hash_g(element, hash_params_g[j][0], hash_params_g[j][1]) for j in range(len(Counter_table)))
            relative_error = abs(count_true - count_estimated) / count_true
            total_relative_error += relative_error
            count += 1
    average_relative_error = total_relative_error / count

    if K <= 20:
        # Sort the histogram by true frequencies in descending order
        sorted_items = sorted(histogram.items(), key=lambda x: x[1], reverse=True)
        for i in range(K):
            element, count_true = sorted_items[i]
            count_estimated = statistics.median(Counter_table[j][hash_f(element, hash_params_f[j][0], hash_params_f[j][1], W)] * hash_g(element, hash_params_g[j][0], hash_params_g[j][1]) for j in range(len(Counter_table)))
            print("Item : ", element, "Freq = ", count_true, " Est. Freq = ", count_estimated)

    print("Avg err for top ", K, "=", average_relative_error)

    F2_normalized = sum(count ** 2 for count in histogram.values()) / (streamLength_filtered[0] ** 2)
    counter_estimated = 0
    for element, count_true in histogram.items():
        counter_estimated += statistics.median(Counter_table[j][hash_f(element, hash_params_f[j][0], hash_params_f[j][1], W)] * hash_g(element, hash_params_g[j][0], hash_params_g[j][1]) for j in range(len(Counter_table)))**2
    F2_approx = counter_estimated/(streamLength_filtered[0] ** 2)
    print("F2", F2_normalized, "F2 Estimate", F2_approx)