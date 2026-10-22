from utils import *
from config import FILE_THOMAS, EPSILON_ZERO, PRINT_PRECISION

def read_thomas_data(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
    n = int(lines[0].strip())
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n
    
    for i in range(n):
        vals = list(map(float, lines[i + 1].split()))
        if i == 0:
            b[0], c[0], d[0] = vals[0], vals[1], vals[2]
        elif i == n - 1:
            a[n - 1], b[n - 1], d[n - 1] = vals[0], vals[1], vals[2]
        else:
            a[i], b[i], c[i], d[i] = vals[0], vals[1], vals[2], vals[3]
            
    return a, b, c, d, n

def thomas_method(a, b, c, d, n):
    alpha = [0.0] * n
    beta = [0.0] * n
    x = [0.0] * n
    
    if abs(b[0]) < EPSILON_ZERO:
        raise ValueError("метод перегонки неприменим")
    
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = a[i] * alpha[i - 1] + b[i]
        if abs(denom) < EPSILON_ZERO:
            raise ValueError("метод перегонки неприменим")
        if i < n - 1:
            alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i] * beta[i - 1]) / denom
        
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
        
    return x, alpha, beta   
         
def run_thomas():
    a, b, c, d, n = read_thomas_data(FILE_THOMAS)
    x, alpha, beta = thomas_method(a, b, c, d, n)
    
    print(f"n = {n}")
    
    print(f"\nalpha = {[round(val, PRINT_PRECISION) for val in alpha]}")
    print(f"\nbeta = {[round(val, PRINT_PRECISION) for val in beta]}")
    print_vector(x, "решение х")
