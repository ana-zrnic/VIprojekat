import networkx as nx
import random
import pandas as pd
import time
import itertools
#from Graph_gen import *
from GraphGen import create_list_of_random_graphs
from GreedyMinIDF import *
from SimulatedAnnealingMinIDF import *
from ACOforMinIDF import *


def func1(arg):
    return greedy_minidf(arg)

def func2(arg):
    return sa_minidf(arg)

def func3(arg):
    return aco_mdr(arg)


def benchmark_function(func, args_list):
    runtimes = []
    results = []
    graph_types = []
    graph_sizes = []
    print(f"Starting benchmark for func**********************************")

    for arg, graph_type, size_label in args_list:
        print(f"Testing {graph_type}, size: {size_label}")
        if func == func2:
            temp_runtimes = []
            temp_results = []

            for _ in range(5):
                start_time = time.time()
                result = func(arg)
                end_time = time.time()
                print(f"Solution for {graph_type}, size: {size_label}: {result}")

                temp_runtimes.append(end_time - start_time)
                temp_results.append(result)

            runtimes.append(temp_runtimes)
            results.append(temp_results)

        else:
            start_time = time.time()
            result = func(arg)
            end_time = time.time()

            runtimes.append(end_time - start_time)
            results.append(result)
        graph_types.append(graph_type)
        graph_sizes.append(size_label)

    print(f"Finished benchmark for func***********************************")
    return runtimes, results, graph_types, graph_sizes


def main():
    args_list = create_list_of_random_graphs()

    functions = [func1, func2, func3]  # Solveri

    all_runtimes = []
    all_results = []
    all_graph_types = []
    all_graph_sizes = []

    for func in functions:
        runtimes, results, all_graph_types, all_graph_sizes = benchmark_function(func, args_list)
        all_runtimes.append(runtimes)
        all_results.append(results)
        #all_graph_types.append(graph_types)
        #all_graph_sizes.append(graph_sizes)

    data = {
        'Graph_Type': all_graph_types,
        'Graph_Size': all_graph_sizes,
        'Func1_Runtime': all_runtimes[0],
        'Func1_Result': all_results[0],        
    }

    for i in range(5):
        data[f"Func2_Runtime_{i + 1}"] = [runtimes[i] for runtimes in all_runtimes[1]]
        data[f"Func2_Result_{i + 1}"] = [result[i] for result in all_results[1]]

    data['Func3_Runtime'] = all_runtimes[2]
    data['Func3_Result'] = all_results[2]

    '''print("size of graph type", len(all_graph_types))
    print("size of graph size", len(all_graph_sizes))
    print("size of graph solution 1", len(all_results[0]))
    print("size of graph time 1", len(all_runtimes[0]))
    print("size of graph solution 2", len(all_results[1]))
    print("size of graph time 2", len(all_runtimes[1]))
    print("size of graph solution 3", len(all_results[2]))
    print("size of graph time 3", len(all_runtimes[2]))'''

    df = pd.DataFrame(data)
    csv_filename = "benchmark_results.csv"
    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    main()