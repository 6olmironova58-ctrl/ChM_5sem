import json
import math

H_DERIVATIVE = 1e-7
H_SECOND_DERIVATIVE = 1e-5

def read_params(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def constants(p):
    return p.get("constants", {})

def vec(lst):
    return [float(x) for x in lst]

def compile_expr(expr_str, variables, consts):
    for k, v in consts.items():
        expr_str = expr_str.replace(k, str(v))
    env = {"__builtins__": {}, "math": math}
    env.update({k: getattr(math, k) for k in dir(math) if not k.startswith('_')})
    env["abs"] = abs
    env["pow"] = pow
    env["ln"] = math.log
    env["lg"] = math.log10
    env["log"] = math.log
    env["log10"] = math.log10
    env["log2"] = math.log2
    
    if len(variables) == 1:
        var = variables[0]
        code = f"lambda {var}: {expr_str}"
        return eval(code, env)
    else:
        var_str = ", ".join(variables)
        code = f"lambda {var_str}: {expr_str}"
        return eval(code, env)

def d1(func, x, h=H_DERIVATIVE):
    return (func(x + h) - func(x - h)) / (2 * h)

def d2(func, x, h=H_SECOND_DERIVATIVE):
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h * h)