import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def plot_equation(f, lo, hi, x0, pts_n, pts_i, root):
    x_vals = np.linspace(lo, hi, 500)
    y_vals = [f(x) for x in x_vals]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    axs[0].plot(x_vals, y_vals, label='f(x)', color='black', linewidth=1.5)
    axs[0].axhline(0, color='black', linewidth=0.5)
    axs[0].axvline(root, color='red', linestyle='--',
                   label=f'Корень: {root:.4f}')

    pts_n_x = [p[0] if isinstance(p, list) else p for p in pts_n]
    pts_n_y = [f(x) for x in pts_n_x]
    axs[0].plot(pts_n_x, pts_n_y, 'o-', color='green',
                label='Итерации Ньютона')

    pts_i_x = [p[0] if isinstance(p, list) else p for p in pts_i]
    pts_i_y = [f(x) for x in pts_i_x]
    axs[0].plot(pts_i_x, pts_i_y, 's-', color='blue',
                label='Метод простой итерации')

    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f(x)')
    axs[0].set_title('Решение нелинейного уравнения')
    axs[0].legend()
    axs[0].grid(True)

    iters_n = range(len(pts_n))
    errs_n = [max(abs((p[0] if isinstance(p, list) else p) - root), 1e-15)
              for p in pts_n]

    iters_i = range(len(pts_i))
    errs_i = [max(abs((p[0] if isinstance(p, list) else p) - root), 1e-15)
              for p in pts_i]

    axs[1].semilogy(iters_n, errs_n, 'o-', color='green',
                    label='Погрешность Ньютона')
    axs[1].semilogy(iters_i, errs_i, 's-', color='blue',
                    label='Погрешность простой итерации')
    axs[1].set_xlabel('Номер итерации')
    axs[1].set_ylabel('Погрешность (лог. шкала)')
    axs[1].set_title('Зависимость погрешности от числа итераций')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_system(f1, f2, x_range, y_range, x0, pts_n, pts_i, root):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    f1_vec = np.vectorize(f1)
    f2_vec = np.vectorize(f2)
    Z1 = f1_vec(X, Y)
    Z2 = f2_vec(X, Y)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].contour(X, Y, Z1, levels=[0], colors='red',
                   linestyles='-', label='f₁ = 0')
    axs[0].contour(X, Y, Z2, levels=[0], colors='blue',
                   linestyles='-', label='f₂ = 0')

    axs[0].plot(x0[0], x0[1], 'ko', label='Начальное приближение')

    pts_n_x = [p[0] for p in pts_n]
    pts_n_y = [p[1] for p in pts_n]
    axs[0].plot(pts_n_x, pts_n_y, 'o-', color='green', label='Метод Ньютона')

    pts_i_x = [p[0] for p in pts_i]
    pts_i_y = [p[1] for p in pts_i]
    axs[0].plot(pts_i_x, pts_i_y, 's-', color='purple',
                label='Метод простой итерации')

    axs[0].plot(root[0], root[1], 'r*', markersize=12,
                label=f'Корень ({root[0]:.3f}, {root[1]:.3f})')

    axs[0].set_xlabel('x₁')
    axs[0].set_ylabel('x₂')
    axs[0].set_title('Решение системы нелинейных уравнений')
    axs[0].legend()
    axs[0].grid(True)

    iters_n = range(len(pts_n))
    errs_n = [max(np.sqrt((p[0] - root[0])**2 + (p[1] - root[1])**2), 1e-15)
              for p in pts_n]

    iters_i = range(len(pts_i))
    errs_i = [max(np.sqrt((p[0] - root[0])**2 + (p[1] - root[1])**2), 1e-15)
              for p in pts_i]

    axs[1].semilogy(iters_n, errs_n, 'o-', color='green',
                    label='Погрешность Ньютона')
    axs[1].semilogy(iters_i, errs_i, 's-', color='purple',
                    label='Погрешность простой итерации')
    axs[1].set_xlabel('Номер итерации')
    axs[1].set_ylabel('Погрешность (лог. шкала)')
    axs[1].set_title('Зависимость погрешности от числа итераций')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    