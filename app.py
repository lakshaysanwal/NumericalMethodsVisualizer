import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

# ================= LOAD CSS =================
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ================= METHODS =================

def bisection(f, a, b, tol, max_iter):
    data = []
    if f(a) * f(b) >= 0:
        return None, None

    for i in range(max_iter):
        c = (a + b) / 2
        error = abs(b - a)
        data.append([i+1, a, b, c, f(c), error])

        if abs(f(c)) < tol or error < tol:
            return c, data

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, data


def newton_raphson(f, df, x0, tol, max_iter):
    data = []
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            return None, None

        x_new = x - fx / dfx
        error = abs(x_new - x)

        data.append([i+1, x, fx, dfx, x_new, error])

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

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x2 - x1)

        data.append([i+1, x0, x1, x2, f(x2), error])

        if error < tol:
            return x2, data

        x0, x1 = x1, x2

    return x2, data


def regula_falsi(f, a, b, tol, max_iter):
    data = []
    if f(a) * f(b) >= 0:
        return None, None

    for i in range(max_iter):
        c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        error = abs(f(c))

        data.append([i+1, a, b, c, f(c), error])

        if abs(f(c)) < tol:
            return c, data

        if f(a)*f(c) < 0:
            b = c
        else:
            a = c

    return c, data


def fixed_point(g, x0, tol, max_iter):
    data = []
    x = x0

    for i in range(max_iter):
        x_new = g(x)
        error = abs(x_new - x)

        data.append([i+1, x, x_new, error])

        if error < tol:
            return x_new, data

        x = x_new

    return x, data


# ================= UI =================

st.sidebar.title("📊 Navigation")

method = st.sidebar.selectbox(
    "Choose Method",
    ["Bisection", "Newton-Raphson", "Secant", "Regula-Falsi", "Fixed Point", "Compare All"]
)

st.markdown("""
<div class="card">
<h1>📊 Numerical Methods Visualizer</h1>
<p>Analyze convergence of numerical algorithms interactively</p>
</div>
""", unsafe_allow_html=True)

# Function input
st.markdown('<div class="card">', unsafe_allow_html=True)

func_input = st.text_input("Enter function f(x):", "x**3 - x - 2")

x = sp.symbols('x')

try:
    f = sp.sympify(func_input)
    f_lambda = sp.lambdify(x, f, "numpy")

    df = sp.diff(f, x)
    df_lambda = sp.lambdify(x, df, "numpy")

    st.latex(f"f(x) = {sp.latex(f)}")

except:
    st.error("Invalid function input!")
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)

# Inputs
if method in ["Bisection", "Regula-Falsi", "Compare All"]:
    a = st.number_input("Enter a:", value=1.0)
    b = st.number_input("Enter b:", value=2.0)

if method in ["Newton-Raphson", "Fixed Point", "Compare All"]:
    x0 = st.number_input("Enter initial guess:", value=1.0)

if method in ["Secant", "Compare All"]:
    x1 = st.number_input("Enter x1:", value=2.0)

tol = st.number_input("Tolerance:", value=0.0001, format="%.5f")
max_iter = st.number_input("Max Iterations:", value=20)

# ================= BUTTON =================

if st.button("Compute Root"):

    if method == "Compare All":

        results = {}
        summary = []

        # ===== BISECTION =====
        root_b, steps_b = bisection(f_lambda, a, b, tol, int(max_iter))
        if steps_b:
            results["Bisection"] = steps_b
            summary.append(["Bisection", root_b, len(steps_b), "Success"])
        else:
            summary.append(["Bisection", "-", "-", "Failed"])

        # ===== NEWTON =====
        root_n, steps_n = newton_raphson(f_lambda, df_lambda, x0, tol, int(max_iter))
        if steps_n:
            results["Newton-Raphson"] = steps_n
            summary.append(["Newton-Raphson", root_n, len(steps_n), "Success"])
        else:
            summary.append(["Newton-Raphson", "-", "-", "Failed"])

        # ===== SECANT =====
        root_s, steps_s = secant(f_lambda, x0, x1, tol, int(max_iter))
        if steps_s:
            results["Secant"] = steps_s
            summary.append(["Secant", root_s, len(steps_s), "Success"])
        else:
            summary.append(["Secant", "-", "-", "Failed"])

        # ===== REGULA FALSI =====
        root_r, steps_r = regula_falsi(f_lambda, a, b, tol, int(max_iter))
        if steps_r:
            results["Regula-Falsi"] = steps_r
            summary.append(["Regula-Falsi", root_r, len(steps_r), "Success"])
        else:
            summary.append(["Regula-Falsi", "-", "-", "Failed"])

        # ===== FIXED POINT =====
        try:
            g = sp.sympify("cos(x)")
            g_lambda = sp.lambdify(x, g, "numpy")
            root_f, steps_f = fixed_point(g_lambda, x0, tol, int(max_iter))

            if steps_f:
                results["Fixed Point"] = steps_f
                summary.append(["Fixed Point", root_f, len(steps_f), "Success"])
            else:
                summary.append(["Fixed Point", "-", "-", "Failed"])
        except:
            summary.append(["Fixed Point", "-", "-", "Error"])

        # ===== DISPLAY SUMMARY =====
        st.subheader("📊 Method Comparison")

        df_summary = pd.DataFrame(summary, columns=["Method", "Root", "Iterations", "Status"])

        tab1, tab2 = st.tabs(["📊 Summary", "📉 Graph"])

        with tab1:
            st.dataframe(df_summary)

        # ===== GRAPH =====
        with tab2:
            fig, ax = plt.subplots()

            for name, steps in results.items():
                errors = [row[-1] for row in steps]
                ax.plot(range(1, len(errors) + 1), errors, marker='o', label=name)

            ax.set_xlabel("Iterations")
            ax.set_ylabel("Error")
            ax.set_title("Comparison of Methods")
            ax.grid(True)
            ax.legend()

            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by: Laksh, Aman & kush")
st.markdown("📊 Numerical Methods Mini Project")