from utils import read_params, vec, compile_expr

DEFAULT_EPS = 0.01
MAX_ITERATIONS = 100
H_DERIVATIVE = 1e-7
JACOBIAN_SINGULAR_THRESHOLD = 0.01
INTERVAL_LOW = -5.0
INTERVAL_HIGH = 5.0

def load(path):
    p = read_params(path)
    n = len(p["x0"])
    vars_list = [f"x{i}" for i in range(n)]
    c = {
        "text": p.get("text", "СНУ"),
        "eps": float(p.get("eps", DEFAULT_EPS)),
        "x0": vec(p["x0"]),
        "n": n,
        "vars": vars_list,
    }
    c["f"] = [compile_expr(fi, vars_list, {}) for fi in p["f"]]
    if "phi" in p and p["phi"]:
        c["phi"] = [compile_expr(pi, vars_list, {}) for pi in p["phi"]]
    else:
        c["phi"] = None
    if "interval" in p:
        c["interval"] = [vec(iv) for iv in p["interval"]]
    else:
        c["interval"] = [[INTERVAL_LOW, INTERVAL_HIGH] for _ in range(n)]
    return c

def jacobian(funcs, x, h=H_DERIVATIVE):
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            x_plus = x[:]
            x_minus = x[:]
            x_plus[j] += h
            x_minus[j] -= h
            J[i][j] = (funcs[i](*x_plus) - funcs[i](*x_minus)) / (2 * h)
    return J

def gauss_solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for k in range(n):
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[max_row][k]) < JACOBIAN_SINGULAR_THRESHOLD:
            raise ValueError("Матрица Якоби вырождена или близка к вырожденной")
        M[k], M[max_row] = M[max_row], M[k]
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= factor * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x

def simple_iteration_system(phi, x0, eps=DEFAULT_EPS, max_iter=MAX_ITERATIONS):
    x = list(x0)
    n = len(x)
    points = [x[:]]
    for _ in range(max_iter):
        x_new = [phi[i](*x) for i in range(n)]
        points.append(x_new[:])
        err = max(abs(x_new[i] - x[i]) for i in range(n))
        if err < eps:
            break
        x = x_new
    return points

def newton_system(f, x0, eps=DEFAULT_EPS, max_iter=MAX_ITERATIONS):
    x = list(x0)
    n = len(x)
    points = [x[:]]
    for _ in range(max_iter):
        F = [f[i](*x) for i in range(n)]
        J = jacobian(f, x)
        b = [-F[i] for i in range(n)]
        dx = gauss_solve(J, b)
        x_new = [x[i] + dx[i] for i in range(n)]
        points.append(x_new[:])
        err = max(abs(dx[i]) for i in range(n))
        if err < eps:
            break
        x = x_new
    return points
