import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



st.set_page_config(page_title="Numerical Methods", layout="wide")

st.markdown("""
<div style="
    background: linear-gradient(90deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
">
    <h1 style="color:#38bdf8;">📊 Numerical Methods Visualizer</h1>
    <p style="color:#9ca3af;">
        Analyze, compare and visualize convergence of numerical root-finding algorithms
    </p>
</div>
""", unsafe_allow_html=True)

from advanced_features import analyze_methods, show_convergence, show_ai_explanation
# ================= FUNCTION =================
def create_function(expr):
    expr = expr.lower().replace("^", "**")

    def f(x):
        return eval(expr, {
            "x": x,
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt
        })
    return f

def derivative(f, x, h=1e-6):
    return (f(x+h) - f(x-h)) / (2*h)

# ================= METHODS =================

def bisection(f, a, b, tol, max_iter):
    data = []
    for i in range(max_iter):
        c = (a + b)/2
        error = abs(b - a)
        data.append([i+1, c, error])

        if abs(f(c)) < tol:
            return c, data

        if f(a)*f(c) < 0:
            b = c
        else:
            a = c

    return c, data


def newton(f, x0, tol, max_iter):
    data = []
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = derivative(f, x)

        if dfx == 0:
            return None, None

        x_new = x - fx/dfx
        error = abs(x_new - x)

        data.append([i+1, x_new, error])

        if error < tol:
            return x_new, data

        x = x_new

    return x, data


def secant(f, x0, x1, tol, max_iter):
    data = []

    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        if fx1 - fx0 == 0:
            return None, None

        x2 = x1 - fx1*(x1-x0)/(fx1-fx0)
        error = abs(x2 - x1)

        data.append([i+1, x2, error])

        if error < tol:
            return x2, data

        x0, x1 = x1, x2

    return x2, data


def regula(f, a, b, tol, max_iter):
    data = []

    for i in range(max_iter):
        c = (a*f(b)-b*f(a))/(f(b)-f(a))
        error = abs(f(c))

        data.append([i+1, c, error])

        if error < tol:
            return c, data

        if f(a)*f(c) < 0:
            b = c
        else:
            a = c

    return c, data


def fixed(g, x0, tol, max_iter):
    data = []
    x = x0

    for i in range(max_iter):
        x_new = g(x)
        error = abs(x_new - x)

        data.append([i+1, x_new, error])

        if error < tol:
            return x_new, data

        x = x_new

    return x, data


# ================= SIDEBAR =================
method = st.sidebar.selectbox("Choose Method", [
    "Bisection", "Newton", "Secant", "Regula", "Fixed Point", "Compare All"
])

# ================= INPUT =================
func_input = st.text_input("Enter f(x):", "x^3 - x - 2")

try:
    f = create_function(func_input)
    f(1)
    st.success("Function OK")
except:
    st.error("Invalid Function")
    st.stop()

a = st.number_input("a", value=1.0)
b = st.number_input("b", value=2.0)
x0 = st.number_input("x0", value=1.0)
x1 = st.number_input("x1", value=2.0)

g_input = st.text_input("g(x) (Fixed Point)", "cos(x)")
g = create_function(g_input)

tol = st.number_input("Tolerance", value=0.0001)
max_iter = st.number_input("Iterations", value=20)

# ================= COMPUTE =================
if st.button("Compute"):

    results = {}
    summary = []

    # ===== SINGLE =====
    if method == "Bisection":
        root, steps = bisection(f, a, b, tol, int(max_iter))
        results["Bisection"] = steps
        summary.append(["Bisection", root])

    elif method == "Newton":
        root, steps = newton(f, x0, tol, int(max_iter))
        results["Newton"] = steps
        summary.append(["Newton", root])

    elif method == "Secant":
        root, steps = secant(f, x0, x1, tol, int(max_iter))
        results["Secant"] = steps
        summary.append(["Secant", root])

    elif method == "Regula":
        root, steps = regula(f, a, b, tol, int(max_iter))
        results["Regula"] = steps
        summary.append(["Regula", root])

    elif method == "Fixed Point":
        root, steps = fixed(g, x0, tol, int(max_iter))
        results["Fixed"] = steps
        summary.append(["Fixed", root])

    # ===== COMPARE =====
    elif method == "Compare All":

        root, steps = bisection(f, a, b, tol, int(max_iter))
        results["Bisection"] = steps
        summary.append(["Bisection", root])

        root, steps = newton(f, x0, tol, int(max_iter))
        results["Newton"] = steps
        summary.append(["Newton", root])

        root, steps = secant(f, x0, x1, tol, int(max_iter))
        results["Secant"] = steps
        summary.append(["Secant", root])

        root, steps = regula(f, a, b, tol, int(max_iter))
        results["Regula"] = steps
        summary.append(["Regula", root])

        root, steps = fixed(g, x0, tol, int(max_iter))
        results["Fixed"] = steps
        summary.append(["Fixed", root])

    # ===== TABLE =====
    st.subheader("Results")
    st.dataframe(pd.DataFrame(summary, columns=["Method", "Root"]))

    # ===== GRAPH =====
    fig, ax = plt.subplots()

    for name, steps in results.items():
        errors = [row[2] for row in steps]
        ax.plot(range(1, len(errors)+1), errors, marker='o', label=name)

    ax.set_title("Convergence Graph")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # ===== ADVANCED FEATURES =====
    if results:
        df_analysis, best = analyze_methods(results)

        if df_analysis is not None:
            st.subheader("📊 Smart Method Analysis")
            st.dataframe(df_analysis)

            st.success(f"🏆 Best Method: {best['Method']}")

            show_convergence(results)

            show_ai_explanation(best)
