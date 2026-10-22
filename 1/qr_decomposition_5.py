import math
from utils import *
from config import FILE_QR, EPSILON_QR, MAX_ITER_QR, EPSILON_ZERO

def qr_householder(A,n):
    Q = identity_matrix(n)
    R = copy_matrix(A)
    cols = [[A[i][j] for i in range(n)] for j in range(n)]
    
    for j in range(n):
        x = [R[i][j] for i in range(j, n)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))

        if norm_x < EPSILON_ZERO:
            continue
        
        v = x[:]
        sign = -1.0 if v[0] >= 0 else 1.0
        v[0] -= sign * norm_x
                
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < EPSILON_ZERO:
            continue
        v = [vi / norm_v for vi in v]
        
        for k in range(j, n):
            dot = sum(v[i - j] * R[i][k] for i in range(j, n))
            for i in range(j, n):
                R[i][k] -= 2.0 * v[i - j] * dot
                
        for i in range(n):
            dot = sum(Q[i][j + k] * v[k] for k in range(n - j))
            for k in range(n - j):
                Q[i][j + k] -= 2.0 * dot * v[k]
                
    return Q, R

def qr_algo(A, n, epsilon=EPSILON_QR, max_iter=MAX_ITER_QR):
    A_k = copy_matrix(A)
    
    for iter in range(max_iter):
        Q, R = qr_householder(A_k, n)
        A_new = multiply_matrices(R, Q)
        
        off_diag_sum = sum(abs(A_new[i][j]) for i in range(n) for j in range(n) if i != j)
        A_k = A_new
        
        if off_diag_sum < epsilon:
            break
        
    eigenvals = []
    i = 0
    while i < n:
        if i == n - 1 or abs(A_k[i+1][i]) < epsilon:
            eigenvals.append(A_k[i][i])
            i += 1
        else:
            tr = A_k[i][i] + A_k[i+1][i+1]
            det = A_k[i][i] * A_k[i+1][i+1] - A_k[i][i+1] * A_k[i+1][i]
            descriminant = tr**2 - 4 * det
            
            if descriminant < 0:
                real_part = tr / 2.0
                imag_part = math.sqrt(-descriminant) / 2.0
                eigenvals.append(complex(real_part, imag_part))
                eigenvals.append(complex(real_part, -imag_part))
            else:
                eigenvals.append((tr + math.sqrt(descriminant)) / 2.0)
                eigenvals.append((tr - math.sqrt(descriminant)) / 2.0)
            i += 2
            
    return eigenvals
    
def run_qr():
    A, n = read_matrix_only(FILE_QR)
    Q, R = qr_householder(A, n)
    eigenvalues = qr_algo(A, n)
    
    print_matrix(A, "матрица А")
    print_matrix(Q, "матрица Q")
    print_matrix(R, "матрица R")
    print_matrix(multiply_matrices(Q, R), "Q * R")
    print("\nСЗ:")
    for i, val in enumerate(eigenvalues):
        if isinstance(val, complex):
            sign = "+" if val.imag >= 0 else "-"
            print(f"  λ[{i}] = {val.real:.6f} {sign} {abs(val.imag):.6f}i")
        else:
            print(f"  λ[{i}] = {val:.8f}")
    
    A_complex = [[2.0, -1.0], [1.0, 2.0]]
    
    import numpy as np
    eigenvalues_np = np.linalg.eigvals(np.array(A_complex))
    print("\nСЗ (NumPy):")
    for val in eigenvalues_np:
        print(f" лямбда = {val}")
