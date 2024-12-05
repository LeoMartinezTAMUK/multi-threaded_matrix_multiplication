# Matrix Multiplication: Sequential vs Parallel Execution

**Author:** Leo Martinez III - [LinkedIn](https://www.linkedin.com/in/leo-martinez-iii/)

**Contact:** [leo.martinez@students.tamuk.edu](mailto:leo.martinez@students.tamuk.edu)

**Created:** Fall 2024

To clone this repository:

```
git clone https://github.com/LeoMartinezTAMUK/multi-threaded_matrix_multiplication.git
```

---

This Python program compares the performance of **sequential** and **parallel** matrix multiplication algorithms. It evaluates the execution times for different matrix sizes and demonstrates the impact of parallelism on computation efficiency.

### Overview:

- **Language**: Python 3.12.6
- **IDE**: Visual Studio Code v1.93.1
- **Libraries Used**:
  - `numpy` (v2.1.2)
  - `threading` (built-in)
  - `multiprocessing` (built-in)
  - `time` (built-in)
  - `csv` (built-in)

### Key Features:

- Implements **sequential** matrix multiplication using NumPy's `np.dot()`.
- Implements **parallel** matrix multiplication using Python's `threading` module.
- Saves the resulting execution times for sequential and parallel algorithms in CSV files.
- Provides insight into the performance trade-offs for matrix multiplication tasks.

### Implementation Details:

- **Sequential Multiplication**:
  - Processes the entire matrix row-by-row.
  - Best for small matrix sizes due to the absence of parallelism overhead.

- **Parallel Multiplication**:
  - Divides matrix rows among multiple threads.
  - Threads compute the dot product for their assigned rows concurrently.
  - Handles load balancing by distributing extra rows evenly among threads.

### Input & Output:

- **Input**: Requires a separate `input.txt` file containing two matrices.
- **Output**:
  - **TerminalOutput.txt**: Logs thread activity and intermediate results.
  - **ResultMatrix.txt**: Saves the final resultant matrix.
  - **Seq_exe.csv**: Contains average sequential execution times.
  - **Paral_exe.csv**: Contains average parallel execution times.

### Usage:

#### Running `matrixmul.py`:
1. Ensure the `input.txt` file with two matrices is in the same directory.
2. Execute the script:
   ```
   python matrixmul.py
   ```
3. Enter the number of threads (`n`):
   - Use `n=1` for sequential execution.
   - Use `n>1` for parallel execution.
4. View the logs in `TerminalOutput.txt` and the results in `ResultMatrix.txt`.

#### Running `matmulperform.py`:
1. Execute the script:
   ```
   python matmulperform.py
   ```
2. Iterates over matrix sizes from 2x2 to 50x50 for 1000 iterations each.
3. Saves execution time results in `Seq_exe.csv` and `Paral_exe.csv`.

### Visualization:

Graphs comparing sequential and parallel execution times are generated from data produced during runtime. These illustrate:
- Overhead for small matrix sizes in parallel execution.
- Efficiency gains for larger matrix sizes with parallelism.

### System Requirements:

- **CPU**: Multi-core processor recommended (tested on a 16-core/32-thread CPU @ 4.2 GHz).
- **Dependencies**: Install `numpy` via pip:
  ```
  pip install numpy
  ```
- **Python Version**: 3.12.6

### Note:

- The program dynamically adjusts the number of threads to prevent idle threads when the matrix size is smaller than the thread count.
- Results may vary based on machine specifications and core/thread availability.

For further details and similar projects, visit my [GitHub Page](https://github.com/LeoMartinezTAMUK).

