import math
from utils import *
from config import FILE_JACOBI, EPSILON_JACOBI, MAX_ITER_JACOBI, EPSILON_ZERO

def jacobi_rotation(A, n, epsilon=EPSILON_JACOBI, max_iter=MAX_ITER_JACOBI):
    A_work = copy_matrix(A)
    V = identity_matrix(n)
    
    for iter in range(max_iter):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_work[i][j]) > max_val:
                    max_val = abs(A_work[i][j])
                    p, q = i, j
                    
        if max_val < epsilon:
            break
        
        if (abs(A_work[p][p] - A_work[q][q])) < EPSILON_ZERO:
            phi = math.pi / 4.0
        else:
            phi = 0.5 * math.atan2(2.0 * A_work[p][q], A_work[p][p] - A_work[q][q])
        
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        new_A = copy_matrix(A_work)
        for i in range(n):
            if i != p and i != q:
                a_ip = A_work[i][p]
                a_iq = A_work[i][q]
                new_A[i][p] = cos_phi * a_ip + sin_phi * a_iq
                new_A[p][i] = new_A[i][p]
                new_A[i][q] = -sin_phi * a_ip + cos_phi * a_iq
                new_A[q][i] = new_A[i][q]
                
        app = A_work[p][p]
        aqq = A_work[q][q]
        apq = A_work[p][q]
        
        new_A[p][p] = cos_phi**2 * app + 2 * sin_phi * cos_phi * apq + sin_phi**2 * aqq
        new_A[q][q] = sin_phi**2 * app - 2 * sin_phi * cos_phi * apq + cos_phi**2 * aqq
        new_A[p][q] = 0.0
        new_A[q][p] = 0.0
        
        A_work = new_A
        
        for i in range(n):
            v_ip = V[i][p]
            v_iq = V[i][q]
            V[i][p] = cos_phi * v_ip + sin_phi * v_iq
            V[i][q] = -sin_phi * v_ip + cos_phi * v_iq
            
    eigenvalues = [A_work[i][i] for i in range(n)]
    return eigenvalues, V
    
def run_jacobi_rotations():
    A, n = read_matrix_only(FILE_JACOBI)
    
    print_matrix(A, "матрица А")
    
    eigenvalues, V = jacobi_rotation(A, n)
    
    print("\nСЗ")
    for i, val in enumerate(eigenvalues):
        print(f"лямбда[{i}] = {val:.8f}")
        
    print_matrix(V, "матрица СВ")
    
    Lambda = create_matrix(n, n)
    for i in range(n):
        Lambda[i][i] = eigenvalues[i]
        
    print_matrix(multiply_matrices(A, V), "A * V")
    print_matrix(multiply_matrices(V, Lambda), "V * лямбда")