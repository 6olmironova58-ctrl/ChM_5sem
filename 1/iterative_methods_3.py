from utils import *
from config import FILE_ITERATIVE, EPSILON_ITERATIVE, MAX_ITER_ITERATIVE
from lu_decomposition_1 import lu_decomposition, solve_system_lu

def jacobi_method(A, b, n, epsilon=EPSILON_ITERATIVE, max_iter=MAX_ITER_ITERATIVE):
    x = [0.0] * n
    iter = 0
    
    for k in range(max_iter):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
            
        iter += 1
        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if diff < epsilon:
            return x_new, iter
        x = x_new
        
    return x, iter
    
def gauss_seidel_method(A, b, n, epsilon=EPSILON_ITERATIVE, max_iter=MAX_ITER_ITERATIVE):
    x = [0.0] * n
    iter = 0
    
    for k in range(max_iter):
        x_old = x[:]
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
            
        iter += 1
        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if diff < epsilon:
            return x, iter
    return x, iter
     
    
def run_iterative():
    A, b, n = read_matrix_from_file(FILE_ITERATIVE)
    print(f"\nэпсилон = {EPSILON_ITERATIVE}")
    
    L, U, P = lu_decomposition(A, n)
    x_exact = solve_system_lu(L, U, P, b, n)
    print_vector(x_exact, "эталон")
    
    x_jacobi, iter_jacobi = jacobi_method(A, b, n)
    print(f"\nЯкоби: {iter_jacobi} итераций")
    print_vector(x_jacobi, "решение")
    
    x_seidel, iter_seidel = gauss_seidel_method(A, b, n)
    print(f"\nЗейдель: {iter_seidel} итераций")
    print_vector(x_seidel, "решение")
    
    print(f"\nСравнение: Якоби - {iter_jacobi}, Зейдель - {iter_seidel}")