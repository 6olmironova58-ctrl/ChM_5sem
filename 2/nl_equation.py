from utils import read_params, vec, compile_expr, constants, d1, d2

DEFAULT_EPS = 1e-4
MAX_ITERATIONS = 100
DERIVATIVE_ZERO_THRESHOLD = 1e-14

def load(path):
    p = read_params(path)
    consts = constants(p)
    c = {"text": p["f"], "eps": float(p.get("eps", DEFAULT_EPS)), "x0": float(p["x0"])}
    c["lo"], c["hi"] = vec(p["interval"])
    c["scan"] = vec(p["scan"]) if "scan" in p else [c["lo"], c["hi"]]
    c["f"] = compile_expr(p["f"], ["x"], consts)
    c["phi"] = compile_expr(p["phi"], ["x"], consts)
    c["df"] = compile_expr(p["df"], ["x"], consts) if "df" in p else (lambda x: d1(c["f"], x))
    c["dphi"] = compile_expr(p["dphi"], ["x"], consts) if "dphi" in p else (lambda x: d1(c["phi"], x))
    c["d2f"] = compile_expr(p["d2f"], ["x"], consts) if "d2f" in p else (lambda x: d2(c["f"], x))
    return c

def simple_iteration_method(phi, dphi, x0, eps=DEFAULT_EPS, max_iter=MAX_ITERATIONS):
    q0 = abs(dphi(x0))
    if q0 >= 1.0:
        raise ValueError(f"Метод итерации: условие |φ'(x)|<1 не выполняется в x0={x0}")
    points = [x0]
    x = x0
    for _ in range(max_iter):
        x_new = phi(x)
        points.append(x_new)
        if abs(x_new - x) < eps:
            break
        x = x_new
    return points

def newton_method(f, df, d2f, x0, eps=DEFAULT_EPS, max_iter=MAX_ITERATIONS):
    if f(x0) * d2f(x0) < 0:
        raise ValueError(f"Метод Ньютона: не выполнено условие f(x0)·f''(x0)>0")
    points = [x0]
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < DERIVATIVE_ZERO_THRESHOLD:
            raise ValueError(f"Метод Ньютона: f'(x)≈0 в точке x={x}")
        x_new = x - fx / dfx
        points.append(x_new)
        if abs(x_new - x) < eps:
            break
        x = x_new
    return points
