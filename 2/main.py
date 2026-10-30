import sys
from utils import read_params
from nl_equation import load as load_eq, newton_method, simple_iteration_method
from nl_system import load as load_sys, newton_system, simple_iteration_system
from plotting import plot_equation, plot_system

def main():
    if len(sys.argv) < 2:
        print("Использование: python main.py <json>")
        return
        
    path = sys.argv[1]
    p = read_params(path)
    
    if isinstance(p["f"], str):
        params = load_eq(path)
        pts_n = newton_method(params["f"], params["df"], params["d2f"], params["x0"], params["eps"])
        pts_i = simple_iteration_method(params["phi"], params["dphi"], params["x0"], params["eps"])
        
        plot_equation(params["f"], params["lo"], params["hi"], params["x0"], pts_n, pts_i, pts_n[-1])
        
    else:
        params = load_sys(path)
        pts_n = newton_system(params["f"], params["x0"], params["eps"])
        pts_i = simple_iteration_system(params["phi"], params["x0"], params["eps"])

        if "interval" in params:
            x_range = params["interval"][0]
            y_range = params["interval"][1]
        else:
            x_range = params.get("x_range", [-1, 5])
            y_range = params.get("y_range", [-1, 5])
            
        plot_system(params["f"][0], params["f"][1], x_range, y_range, params["x0"], pts_n, pts_i, pts_n[-1])

if __name__ == "__main__":
    main()