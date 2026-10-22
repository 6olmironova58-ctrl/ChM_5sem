from lu_decomposition_1 import run_lu
from thomas_method_2 import run_thomas
from iterative_methods_3 import run_iterative
from jacobi_rotations_4 import run_jacobi_rotations
from qr_decomposition_5 import run_qr

def main():
    print("1. LU-разложение")
    run_lu()
    print("\n2. Метод прогонки")
    run_thomas()
    print("\n3. Итерационные методы")
    run_iterative()
    print("\n4. Метод вращений Якоби")
    run_jacobi_rotations()
    print("\n5. QR-разложение")
    run_qr()

if __name__ == '__main__':
    main()