from utils import *
from config import FILE_LU, EPSILON_ZERO, PRINT_PRECISION

def lu_decomposition(A, n):
    U = copy_matrix(A)
    L = identity_matrix(n)
    P = identity_matrix(n)
    for k in range(n):
        max_val = abs(U[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i
                
        if max_val < EPSILON_ZERO:
           raise ValueError("матрица вырождена")
                
        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            P[k], P[max_row] = P[max_row], P[k]
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
                
        for i in range(k+1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
                
    return L, U, P
    
def forward_substitution(L, b, n):
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - s) / L[i][i]
    return y
    
def back_substitution(U, y, n):
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]
    return x
    
def solve_system_lu(L, U, P, b, n):
    Pb = multiply_matrix_vector(P, b)
    y = forward_substitution(L, Pb, n)
    x = back_substitution(U, y, n)
    return x

def find_inverse_lu(L, U, P, n):
    E = identity_matrix(n)
    inv_A = create_matrix(n, n)
    for j in range(n):
        ej = [E[i][j] for i in range(n)]
        xj = solve_system_lu(L, U, P, ej, n)
        for i in range(n):
            inv_A[i][j] = xj[i]
            
    return inv_A
    
def determinant_lu(U, P, n):
    visited = [False] * n
    det_P = 1.0
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                for k in range(n):
                    if P[j][k] == 1.0:
                        j = k
                        break
                cycle_len += 1
            if (cycle_len - 1) % 2 == 1:
                det_P *= -1.0
                
    det_U = 1.0
    for i in range(n):
        det_U *= U[i][i]
    
    return det_U / det_P
    
def run_lu():
    A, b, n = read_matrix_from_file(FILE_LU)
    
    print_matrix(A, "матрица А")
    print_vector(b, "вектор b")
    
    L, U, P = lu_decomposition(A, n)
    
    print_matrix(L, "матрица L")
    print_matrix(U, "матрица U")
    print_matrix(multiply_matrices(L, U), "L * U")
    
    x = solve_system_lu(L, U, P, b, n)
    
    print_vector(x, "решение х")
    
    inv_A = find_inverse_lu(L, U, P, n)
    det = determinant_lu(U, P, n)
    
    print_matrix(inv_A, "матрица А^(-1)")
    print(f"\ndet(A)) = {det:.{PRINT_PRECISION}f}")
    
    print_matrix(multiply_matrices(A, inv_A), "A * A^(-1)")
    print_matrix(multiply_matrices(P, A), "P*A")
    