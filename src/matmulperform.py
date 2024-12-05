# Sequential and Parallel Execution Performance Compare of Matrix Multiplication
# Created by: Leo Martinez III in Fall 2024

import numpy as np
import threading # POSIX
import multiprocessing # for counting cores on the machine
import time
import csv

# Function to generate random matrices of size n x n
def generate_random_matrix(n):
    return np.random.uniform(-10, 10, (n, n))  # generates a matrix with values between -10 and 10

# Function for sequential matrix multiplication
def sequential_multiply(matrix_a, matrix_b):
    return np.dot(matrix_a, matrix_b)

# Function for parallel matrix multiplication (work division)
def parallel_multiply(matrix_a, matrix_b, n_threads):
    num_rows = matrix_a.shape[0]
    num_cols = matrix_b.shape[1]
    result = np.zeros((num_rows, num_cols))  # initialize with all zeros

    # Adjust n_threads if there are more threads than rows
    if n_threads > num_rows:
        n_threads = num_rows

    def worker(thread_id, start_row, end_row):
        for i in range(start_row, end_row):
            result[i] = np.dot(matrix_a[i], matrix_b)
    
    threads = []
    chunk_size = num_rows // n_threads  # each thread is assigned a portion of the matrix rows
    extra_rows = num_rows % n_threads  # handle remaining rows when division is not even

    start_row = 0
    for i in range(n_threads):
        end_row = start_row + chunk_size + (1 if i < extra_rows else 0)  # distribute extra rows evenly
        thread = threading.Thread(target=worker, args=(i, start_row, end_row))
        threads.append(thread)
        thread.start()
        start_row = end_row

    for thread in threads:
        thread.join()  # ensure all threads finish before returning the result

    return result

# Function to measure average execution time for sequential multiplication
def measure_sequential(n, iterations):
    times = []
    for _ in range(iterations): # underscore is just a placeholder
        matrix_a = generate_random_matrix(n)
        matrix_b = generate_random_matrix(n)
        start_time = time.perf_counter()  # start time
        sequential_multiply(matrix_a, matrix_b)
        end_time = time.perf_counter()  # end time
        times.append(end_time - start_time)
    return sum(times) / iterations  # return the mean (average) time

# Function to measure average execution time for parallel multiplication
def measure_parallel(n, iterations, n_threads):
    times = []
    for _ in range(iterations):
        matrix_a = generate_random_matrix(n)
        matrix_b = generate_random_matrix(n)
        start_time = time.perf_counter() # start time
        parallel_multiply(matrix_a, matrix_b, n_threads)
        end_time = time.perf_counter()  # end time
        times.append(end_time - start_time)
    return sum(times) / iterations  # return the mean (average) time

# Main function to compute and store execution times in CSV files
def main():
    iterations = 1000  # minimum 100 iterations for each n (value can be changed as needed)
    seq_times = []
    par_times = []

    # Loop through matrix sizes n from 2 to 50
    for n in range(2, 51):
        n_threads = min(n, multiprocessing.cpu_count()) # to avoid using more cores than available
        print(f"Processing matrix size n = {n}...")

        # Measure sequential execution time
        seq_avg_time = measure_sequential(n, iterations)
        seq_times.append([n, seq_avg_time])
        print(f"Sequential: n={n}, avg_time={seq_avg_time:.6f} seconds")

        # Measure parallel execution time with n threads
        par_avg_time = measure_parallel(n, iterations, n_threads)
        par_times.append([n, par_avg_time])
        print(f"Parallel: n={n}, avg_time={par_avg_time:.6f} seconds")

    # Write results to CSV files
    with open('Seq_exe.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'avg_time'])
        writer.writerows(seq_times)

    with open('Paral_exe.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n', 'avg_time'])
        writer.writerows(par_times)

    print("Execution times saved to 'Seq_exe.csv' and 'Paral_exe.csv'.")

# Executes the main function if this file is run directly
if __name__ == '__main__':
    main()

