from config import PRINT_PRECISION, PRINT_WIDTH

def create_matrix(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def identity_matrix(n):
    mat = create_matrix(n, n)
    for i in range(n):
        mat[i][i] = 1.0
    return mat

def copy_matrix(mat):
    return [row[:] for row in mat]

def print_matrix(mat, name=""):
    if name:
        print(f"\n{name}:")
    for row in mat:
        print("  ".join(f"{val:{PRINT_WIDTH}.{PRINT_PRECISION}f}" for val in row))

def print_vector(vec, name=""):
    if name:
        print(f"\n{name}:")
    for val in vec:
        print(f"  {val:.{PRINT_PRECISION}f}")

def multiply_matrices(a, b):
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = create_matrix(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result

def multiply_matrix_vector(a, v):
    rows = len(a)
    cols = len(a[0])
    result = [0.0] * rows
    for i in range(rows):
        for j in range(cols):
            result[i] += a[i][j] * v[j]
    return result

def read_matrix_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
    n = int(lines[0].strip())
    A = create_matrix(n, n)
    b = [0.0] * n
    for i in range(n):
        values = list(map(float, lines[i + 1].strip().split()))
        for j in range(n):
            A[i][j] = values[j]
        b[i] = values[n]
    return A, b, n

def read_matrix_only(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
    n = int(lines[0].strip())
    A = create_matrix(n, n)
    for i in range(n):
        values = list(map(float, lines[i + 1].strip().split()))
        for j in range(n):
            A[i][j] = values[j]
    return A, n