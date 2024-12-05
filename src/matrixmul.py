# Sequential or Parallel Execution of Matrix Multiplication
# Created by: Leo Martinez III in Fall 2024

import numpy as np # matrices support
import threading # POSIX

# Function to read matrices from input.txt
def read_matrices(filename):
    with open(filename, 'r') as f: # 'r' stands for read
        data = f.read().strip().split('\n\n') # double new line indicates new matrix
        matrix_a = np.array([[float(num) for num in row.split(',')] for row in data[0].splitlines()])
        matrix_b = np.array([[float(num) for num in row.split(',')] for row in data[1].splitlines()])
    return matrix_a, matrix_b

# Function for sequential matrix multiplication
def sequential_multiply(matrix_a, matrix_b):
    return np.dot(matrix_a, matrix_b)

# Function for parallel matrix multiplication (work division)
def parallel_multiply(matrix_a, matrix_b, n_threads, log_file):
    num_rows = matrix_a.shape[0]
    num_cols = matrix_b.shape[1]
    result = np.zeros((num_rows, num_cols)) # initialize with all zeros

    # Adjust n_threads if there are more threads than rows (added later on)
    if n_threads > num_rows:
        n_threads = num_rows

    def worker(thread_id, start_row, end_row):
        for i in range(start_row, end_row):
            result[i] = np.dot(matrix_a[i], matrix_b)
            log_file.write(f"Thread {thread_id}: Computed row {i} => Intermediate result: {result[i]}\n")
            print(f"Thread {thread_id}: Computed row {i} => Intermediate result: {result[i]}")  # Print to console
    
    threads = []
    chunk_size = num_rows // n_threads # each thread is assigned a portion of the matrix rows to compute
    extra_rows = num_rows % n_threads  # handle the case where rows are not evenly divisible (ex: 3 rows when n = 4)

    start_row = 0
    for i in range(n_threads):
        end_row = start_row + chunk_size + (1 if i < extra_rows else 0)  # distribute extra rows evenly
        log_file.write(f"Thread {i} will compute rows {start_row} to {end_row - 1}\n")
        print(f"Thread {i} will compute rows {start_row} to {end_row - 1}")  # print to console
        thread = threading.Thread(target=worker, args=(i, start_row, end_row))
        threads.append(thread)
        thread.start()
        start_row = end_row

    for thread in threads: # this loop ensures that the main thread waits for all worker threads to finish execution before proceeding
        thread.join()

    return result

# Main function
def main():
    matrix_a, matrix_b = read_matrices('input.txt') # input file should already have data in it
    n_threads = int(input("Enter the number of threads (lightweight processes) (n): ")) # n=1 for sequential, n>1 for parallel processing
    
    # Check matrix size limits (Not required to function) (100 is the max size)
    if matrix_a.shape[0] > 100 or matrix_a.shape[1] > 100 or matrix_b.shape[0] > 100 or matrix_b.shape[1] > 100:
        print("Error: The maximum size of a row or a column in a matrix can be 100.")
        return

    with open('TerminalOutput.txt', 'w') as log_file: # note: 'w' (write) will overwrite any previous results, 'a' (append) can be used instead if preferred
        if n_threads == 1:
            log_file.write("Performing sequential matrix multiplication... (n=1)\n")
            result = sequential_multiply(matrix_a, matrix_b)
            log_file.write("Resultant Matrix:\n")
            log_file.write(f"{result}\n")
            print("Resultant Matrix:\n")
            print(result)  # print to terminal
        else:
            log_file.write(f"Performing parallel matrix multiplication with {n_threads} threads... (n>1)\n")
            result = parallel_multiply(matrix_a, matrix_b, n_threads, log_file)
            log_file.write("Resultant Matrix:\n")
            log_file.write(f"{result}\n")
            print("Resultant Matrix:\n")
            print(result)  # print to terminal

    np.savetxt('ResultMatrix.txt', result, fmt='%.2f')

# Executes the main() function only if the script is run directly, not when imported as a module (Not required to function)
if __name__ == '__main__':
    main()
