import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import time
import itertools
import csv
from tabulate import tabulate
from typing import List, Tuple, Any
import hnswlib
import argparse
from tqdm import tqdm
import signal
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Signal handler setup
def signal_handler(sig, frame):
    print('\nInterrupted by user. Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Compare KNN search algorithms")
    parser.add_argument("--vectors", nargs="+", type=int, default=[1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000],
                        help="List of vector counts to test")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[4, 16, 256, 1024],
                        help="List of dimensions to test")
    parser.add_argument("--num-tests", type=int, default=10,
                        help="Number of tests to run for each combination")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of nearest neighbors to search for")
    args = parser.parse_args()

    print("Starting tests for all combinations...")

    results = []
    headers = ["Num Vectors", "Num Dimensions", "KD-Tree Build Time", "Ball Tree Build Time", "HNSW Build Time", 
               "KD-Tree Search Time", "Ball Tree Search Time", "Brute Force Search Time", "HNSW Search Time"]

    total_combinations = len(args.vectors) * len(args.dimensions)
    try:
        for num_vectors, num_dimensions in tqdm(itertools.product(args.vectors, args.dimensions), total=total_combinations, desc="Testing combinations"):
            print(f"\nTesting with {num_vectors} vectors of {num_dimensions} dimensions")
            print(f"Running {args.num_tests} tests, searching for {args.k} nearest neighbors each time")

            vectors = np.random.uniform(-1, 1, (num_vectors, num_dimensions))
            
            print("Building KD-Tree...")
            kd_tree_start = time.time()
            kd_tree = KDTree(vectors)
            kd_tree_build_time = time.time() - kd_tree_start

            print("Building Ball Tree...")
            ball_tree_start = time.time()
            ball_tree = BallTree(vectors)
            ball_tree_build_time = time.time() - ball_tree_start

            print("Building HNSW index...")
            hnsw_start = time.time()
            hnsw_index = build_hnsw_index(vectors, num_dimensions, num_vectors)
            hnsw_build_time = time.time() - hnsw_start

            if hnsw_index is None:
                print("Skipping HNSW tests due to index building failure.")
                hnsw_build_time = float('nan')
                avg_hnsw_time = float('nan')
            else:
                kd_times, ball_times, brute_times, hnsw_times = [], [], [], []

                for _ in tqdm(range(args.num_tests), desc="Running tests", leave=False):
                    query_vector = np.random.uniform(-1, 1, num_dimensions)
                    
                    kd_time, kd_neighbors, kd_distances = run_test(kdtree_knn, kd_tree, query_vector, args.k)
                    ball_time, ball_neighbors, ball_distances = run_test(balltree_knn, ball_tree, query_vector, args.k)
                    brute_time, brute_neighbors, brute_distances = run_test(brute_force_knn, vectors, query_vector, args.k)
                    hnsw_time, hnsw_neighbors, hnsw_distances = run_test(hnsw_knn, hnsw_index, query_vector, args.k)
                    
                    kd_times.append(kd_time)
                    ball_times.append(ball_time)
                    brute_times.append(brute_time)
                    hnsw_times.append(hnsw_time)
                    
                    results_match, match_details = compare_results(kd_distances, ball_distances, brute_distances, hnsw_distances)
                    if not results_match:
                        print(f"\nMismatch in test: {match_details}")

                avg_kd_time = np.mean(kd_times)
                avg_ball_time = np.mean(ball_times)
                avg_brute_time = np.mean(brute_times)
                avg_hnsw_time = np.mean(hnsw_times)

            results.append([num_vectors, num_dimensions, 
                            kd_tree_build_time, ball_tree_build_time, hnsw_build_time,
                            avg_kd_time, avg_ball_time, avg_brute_time, avg_hnsw_time])

            print(f"\nResults for {num_vectors} vectors with {num_dimensions} dimensions:")
            print(f"KD-Tree build time:       {kd_tree_build_time:.6f} seconds")
            print(f"Ball Tree build time:     {ball_tree_build_time:.6f} seconds")
            print(f"HNSW build time:          {hnsw_build_time:.6f} seconds")
            print(f"KD-Tree search time:      {avg_kd_time:.6f} seconds")
            print(f"Ball Tree search time:    {avg_ball_time:.6f} seconds")
            print(f"Brute Force search time:  {avg_brute_time:.6f} seconds")
            print(f"HNSW search time:         {avg_hnsw_time:.6f} seconds")

    except KeyboardInterrupt:
        print('\nInterrupted by user. Exiting gracefully...')
        sys.exit(0)

    print("\nAll tests completed.")

    # Print results as a table
    print("\nResults Table:")
    print(tabulate(results, headers=headers, tablefmt="grid"))

    # Save results to CSV
    csv_filename = "knn_search_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(results)

    print(f"\nResults saved to {csv_filename}")

    # Create and save the chart
    create_results_chart(csv_filename)

# KNN search functions
def brute_force_knn(vectors: np.ndarray, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a KNN search using brute force.

    Args:
        vectors (np.ndarray): The dataset vectors.
        query_vector (np.ndarray): The query vector.
        k (int): The number of nearest neighbors to find.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the k nearest neighbors and their distances.
    """
    distances = np.linalg.norm(vectors - query_vector, axis=1)
    nearest_indices = np.argpartition(distances, k)[:k]
    sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]
    return vectors[sorted_indices], distances[sorted_indices]

def kdtree_knn(tree: KDTree, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a KNN search using KD-Tree.

    Args:
        tree (KDTree): The KD-Tree to search.
        query_vector (np.ndarray): The query vector.
        k (int): The number of nearest neighbors to find.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the k nearest neighbors and their distances.
    """
    distances, indices = tree.query(query_vector.reshape(1, -1), k=k)
    return tree.data[indices[0]], distances[0]

def balltree_knn(tree: BallTree, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a KNN search using Ball Tree.

    Args:
        tree (BallTree): The Ball Tree to search.
        query_vector (np.ndarray): The query vector.
        k (int): The number of nearest neighbors to find.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the k nearest neighbors and their distances.
    """
    distances, indices = tree.query(query_vector.reshape(1, -1), k=k)
    indices = indices[0]
    distances = distances[0]
    return tree.get_arrays()[0][indices], distances

def hnsw_knn(index: hnswlib.Index, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a KNN search using HNSW.

    Args:
        index (hnswlib.Index): The HNSW index to search.
        query_vector (np.ndarray): The query vector.
        k (int): The number of nearest neighbors to find.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the k nearest neighbors and their distances.
    """
    labels, distances = index.knn_query(query_vector.reshape(1, -1), k=k)
    neighbors = index.get_items(labels[0])
    return neighbors, np.sqrt(distances[0])  # Take square root of distances

# Helper functions
def build_hnsw_index(vectors, num_dimensions, num_vectors):
    try:
        hnsw_index = hnswlib.Index(space='l2', dim=num_dimensions)
        
        # Adjust these parameters
        ef_construction = min(100, num_vectors)  # Limit ef_construction based on vector count
        M = min(16, num_dimensions)  # Limit M based on dimensions
        
        hnsw_index.init_index(max_elements=num_vectors, ef_construction=ef_construction, M=M)
        
        ef_search = min(50, num_vectors)  # Limit ef_search based on vector count
        hnsw_index.set_ef(ef_search)
        
        batch_size = min(1000, num_vectors // 10)  # Adjust batch size based on total vectors
        for i in range(0, num_vectors, batch_size):
            batch = vectors[i:i+batch_size]
            hnsw_index.add_items(batch, np.arange(i, i+len(batch)))
            
            # Check for interruption
            if signal.SIGINT.value in signal.sigpending():
                print('\nInterrupted during HNSW index building. Exiting gracefully...')
                sys.exit(0)
        
        return hnsw_index
    except MemoryError:
        print(f"Memory error while building HNSW index for {num_vectors} vectors with {num_dimensions} dimensions.")
        return None
    except Exception as e:
        print(f"Error building HNSW index: {e}")
        return None

def compare_results(kd_tree_results, ball_tree_results, brute_force_results, hnsw_results):
    """
    Compare the results from different algorithms.
    Returns a tuple with a boolean indicating if results match and a string with details.
    """
    def sort_results(vectors, distances):
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            return sorted(zip(map(tuple, vectors), distances), key=lambda x: (x[1], x[0]))
        elif isinstance(vectors, np.ndarray) and vectors.ndim == 1:
            return [(tuple(vectors), distances)]
        else:
            return [(vectors, distances)]

    kd_sorted = sort_results(kd_tree_results[0], kd_tree_results[1])
    ball_sorted = sort_results(ball_tree_results[0], ball_tree_results[1])
    brute_sorted = sort_results(brute_force_results[0], brute_force_results[1])
    hnsw_sorted = sort_results(hnsw_results[0], hnsw_results[1])
    
    def get_distances(sorted_results):
        return [x[1] for x in sorted_results]

    kd_distances = get_distances(kd_sorted)
    ball_distances = get_distances(ball_sorted)
    brute_distances = get_distances(brute_sorted)
    hnsw_distances = get_distances(hnsw_sorted)

    exact_match = np.allclose(kd_distances, ball_distances) and np.allclose(kd_distances, brute_distances)
    
    hnsw_match_percentage = np.mean([np.isclose(hnsw_dist, kd_dist, rtol=1e-2, atol=1e-2) 
                                     for hnsw_dist, kd_dist in zip(hnsw_distances, kd_distances)])
    hnsw_threshold = 0.90  # Lower the threshold to 90%

    all_match = exact_match and (hnsw_match_percentage >= hnsw_threshold)
    
    details = []
    if not exact_match:
        if not np.allclose(kd_distances, ball_distances):
            details.append(f"KD-Tree and Ball Tree results differ: KD={kd_sorted[:3]}, Ball={ball_sorted[:3]}")
        if not np.allclose(kd_distances, brute_distances):
            details.append(f"KD-Tree and Brute Force results differ: KD={kd_sorted[:3]}, Brute={brute_sorted[:3]}")
        if not np.allclose(ball_distances, brute_distances):
            details.append(f"Ball Tree and Brute Force results differ: Ball={ball_sorted[:3]}, Brute={brute_sorted[:3]}")
    
    if hnsw_match_percentage < hnsw_threshold:
        details.append(f"HNSW match percentage ({hnsw_match_percentage:.2%}) below threshold ({hnsw_threshold:.2%})")
        details.append(f"HNSW results: {hnsw_sorted[:3]}")
    
    detail_str = "; ".join(details) if details else "All results match"
    
    return all_match, detail_str

def run_test(search_func: callable, *args: Any) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run a single test for a given search function.

    Args:
        search_func (callable): The search function to test.
        *args: Arguments to pass to the search function.

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: A tuple containing the execution time,
        the nearest neighbors, and their distances.
    """
    start_time = time.time()
    neighbors, distances = search_func(*args)
    execution_time = time.time() - start_time
    
    # Ensure neighbors and distances are always arrays
    if not isinstance(neighbors, np.ndarray):
        neighbors = np.array([neighbors])
    if not isinstance(distances, np.ndarray):
        distances = np.array([distances])
    
    return execution_time, neighbors, distances

def create_results_chart(csv_filename):
    # Read the CSV file
    df = pd.read_csv(csv_filename)

    # Create subplots for each dimension
    dimensions = df['Num Dimensions'].unique()
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(15, 5*len(dimensions)), sharex=True)
    fig.suptitle('KNN Search Algorithm Comparison', fontsize=16)

    # Color palette for algorithms
    colors = {'KD-Tree': 'blue', 'Ball Tree': 'orange', 'Brute Force': 'green', 'HNSW': 'red'}

    for i, dim in enumerate(dimensions):
        dim_data = df[df['Num Dimensions'] == dim]
        
        for algo in ['KD-Tree', 'Ball Tree', 'Brute Force', 'HNSW']:
            axes[i].plot(dim_data['Num Vectors'], dim_data[f'{algo} Search Time'], 
                         marker='o', label=algo, color=colors[algo])
        
        axes[i].set_ylabel('Search Time (seconds)')
        axes[i].set_yscale('log')
        axes[i].set_title(f'Dimensions: {dim}')
        axes[i].grid(True, which="both", ls="-", alpha=0.2)
        axes[i].legend()

    # Set x-axis labels on the bottom subplot
    axes[-1].set_xlabel('Number of Vectors')
    axes[-1].set_xticks(df['Num Vectors'].unique())
    axes[-1].set_xticklabels(df['Num Vectors'].unique(), rotation=45)

    plt.tight_layout()
    plt.savefig('knn_search_comparison.png', dpi=300, bbox_inches='tight')
    print("Chart saved as knn_search_comparison.png")
    plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    main()