import os
import math
import numpy as np
import matplotlib.pyplot as plt

SAVE_FIGS = True
a, b = 0.0, 24.0
EPS = 1e-12


# ----------------------------------------------------------------------
# Підінтегральна функція (навантаження на сервер)
# ----------------------------------------------------------------------
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


def show_figure(fig, name):
    fig.tight_layout()
    if SAVE_FIGS:
        os.makedirs("graphs", exist_ok=True)
        fig.savefig(os.path.join("graphs", name), dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# 2. Точне значення інтеграла
#    int 50 dx = 50*(b-a);  int 20 sin(pi x/12) dx = 0 на [0, 24] (повний період);
#    int 5 exp(-0.2 (x-12)^2) dx = 5*sqrt(pi/0.2)*erf(12*sqrt(0.2))
# ----------------------------------------------------------------------
I0_exact = 50 * (b - a) + 0.0 + 5 * math.sqrt(math.pi / 0.2) * math.erf(12 * math.sqrt(0.2))
print("2. Точне значення I0 = %.12f" % I0_exact)


# ----------------------------------------------------------------------
# 3. Складова формула Сімпсона (N - парне)
#    I(N) = h/3 * (f0 + 4(f1+f3+...+f_{N-1}) + 2(f2+f4+...+f_{N-2}) + fN)
# ----------------------------------------------------------------------
def simpson(N):
    if N % 2:
        raise ValueError("N має бути парним")
    h = (b - a) / N
    x = a + h * np.arange(N + 1)
    y = f(x)
    return h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum() + y[-1])


# ----------------------------------------------------------------------
# 4. Залежність точності від N (N = 10...1000), пошук N_opt
# ----------------------------------------------------------------------
Ns = np.arange(10, 1001, 2)
errs = np.array([abs(simpson(int(N)) - I0_exact) for N in Ns])

idx = np.where(errs <= EPS)[0]
N_opt = int(Ns[idx[0]]) if len(idx) else int(Ns[np.argmin(errs)])
epsopt = abs(simpson(N_opt) - I0_exact)
print("\n4. Задана точність eps = %.0e" % EPS)
print("   N_opt = %d,  epsopt = %.3e" % (N_opt, epsopt))

# ----------------------------------------------------------------------
# 5. N0 ~ N_opt/10, кратне 8
# ----------------------------------------------------------------------
N0 = max(8, int(round(N_opt / 10 / 8)) * 8)
I_N0 = simpson(N0)
eps0 = abs(I_N0 - I0_exact)
print("\n5. N0 = %d,  I(N0) = %.12f,  eps0 = %.3e" % (N0, I_N0, eps0))

# ----------------------------------------------------------------------
# 6. Метод Рунге-Ромберга (p = 4:  2^p - 1 = 15)
# ----------------------------------------------------------------------
I_h2 = simpson(N0 // 2)
I_R = I_N0 + (I_N0 - I_h2) / 15
epsR = abs(I_R - I0_exact)
print("\n6. Рунге-Ромберг: I(N0/2) = %.12f" % I_h2)
print("   I_R = %.12f,  epsR = %.3e" % (I_R, epsR))

# ----------------------------------------------------------------------
# 7. Метод Ейткена (N0, N0/2, N0/4)
# ----------------------------------------------------------------------
I_h4 = simpson(N0 // 4)
I_E = (I_h2 ** 2 - I_N0 * I_h4) / (2 * I_h2 - (I_N0 + I_h4))
p = math.log(abs((I_h4 - I_h2) / (I_h2 - I_N0))) / math.log(2)
epsE = abs(I_E - I0_exact)
print("\n7. Ейткен: I(N0/4) = %.12f" % I_h4)
print("   I_E = %.12f,  p = %.4f,  epsE = %.3e" % (I_E, p, epsE))

print("\n8. Похибки: eps0 = %.2e;  epsR = %.2e;  epsE = %.2e" % (eps0, epsR, epsE))

# Додаткове дослідження: поведінка методів при різних N0 (кратних 8)
print("\n8*. Похибки методів залежно від N0")
print("   %-5s %-11s %-11s %-11s %-8s" % ("N0", "eps0", "epsR", "epsE", "p"))
N0_list = list(range(8, 97, 8))
tab = []
for n0 in N0_list:
    i0, i1, i2 = simpson(n0), simpson(n0 // 2), simpson(n0 // 4)
    iR = i0 + (i0 - i1) / 15
    try:
        iE = (i1 ** 2 - i0 * i2) / (2 * i1 - (i0 + i2))
        pp = math.log(abs((i2 - i1) / (i1 - i0))) / math.log(2)
    except (ZeroDivisionError, ValueError):
        iE, pp = float("nan"), float("nan")
    row = (n0, abs(i0 - I0_exact), abs(iR - I0_exact), abs(iE - I0_exact), pp)
    tab.append(row)
    print("   %-5d %-11.2e %-11.2e %-11.2e %-8.2f" % row)


# ----------------------------------------------------------------------
# 9. Адаптивний алгоритм на основі формули Сімпсона
# ----------------------------------------------------------------------
class Counter:
    """Лічильник обчислень підінтегральної функції."""
    n = 0


def f_count(x):
    Counter.n += 1
    return float(f(x))


def adaptive(x0, x1, f0, fm, f1, delta, total_len):
    """Один крок адаптивного алгоритму на відрізку [x0, x1].
    I1 - Сімпсон з одним відрізком, I2 - Сімпсон після ділення навпіл.
    Умова збіжності (3): |I1 - I2| <= delta_i,
    де delta_i = delta * (x1 - x0) / (b - a) (пропорційно довжині відрізка)."""
    h = x1 - x0
    xm = x0 + h / 2
    f_l = f_count(x0 + h / 4)
    f_r = f_count(x0 + 3 * h / 4)
    I1 = h / 6 * (f0 + 4 * fm + f1)
    I2 = h / 12 * (f0 + 4 * f_l + fm) + h / 12 * (fm + 4 * f_r + f1)
    delta_i = delta * h / total_len
    if abs(I1 - I2) <= delta_i:
        return I2
    return (adaptive(x0, xm, f0, f_l, fm, delta, total_len) +
            adaptive(xm, x1, fm, f_r, f1, delta, total_len))


def adaptive_integral(delta):
    Counter.n = 0
    f0, f1, fm = f_count(a), f_count(b), f_count((a + b) / 2)
    I = adaptive(a, b, f0, fm, f1, delta, b - a)
    return I, Counter.n


print("\n9. Адаптивний алгоритм")
print("   %-10s %-18s %-12s %-10s" % ("delta", "I", "eps", "к-сть f"))
deltas = np.array([10.0 ** (-k) for k in range(1, 13)])
eps_ad, evals_ad = [], []
for d in deltas:
    I_ad, n_ev = adaptive_integral(d)
    e = abs(I_ad - I0_exact)
    eps_ad.append(e)
    evals_ad.append(n_ev)
    print("   %-10.0e %-18.12f %-12.3e %-10d" % (d, I_ad, e, n_ev))


# ----------------------------------------------------------------------
# ГРАФІКИ
# ----------------------------------------------------------------------
# Рис.1. Підінтегральна функція
xx = np.linspace(a, b, 600)
fig1 = plt.figure("Рис.1 – Навантаження на сервер", figsize=(8, 4.5))
plt.plot(xx, f(xx), label=r"$f(x)=50+20\sin(\pi x/12)+5e^{-0.2(x-12)^2}$")
plt.title("Навантаження на сервер f(x)")
plt.xlabel("Час, x (год)")
plt.ylabel("Навантаження, f(x)")
plt.legend()
plt.grid(True)
show_figure(fig1, "fig1_function.png")

# Рис.2. Залежність похибки від N
fig2 = plt.figure("Рис.2 – Залежність похибки від N", figsize=(8, 4.5))
plt.semilogy(Ns, np.where(errs > 0, errs, np.nan), label="eps(N)")
plt.axhline(EPS, color="r", linestyle="--", label="eps = 1e-12")
plt.plot(N_opt, epsopt, "ro", label="N_opt = %d" % N_opt)
plt.title("Залежність похибки складової формули Сімпсона від N")
plt.xlabel("N")
plt.ylabel("eps(N) = |I(N) - I0|")
plt.legend()
plt.grid(True, which="both")
show_figure(fig2, "fig2_error_vs_N.png")

# Рис.3. Порівняння похибок методів
fig3 = plt.figure("Рис.3 – Порівняння методів", figsize=(8, 4.5))
plt.bar(["eps0 (Сімпсон)", "epsR (Рунге-Ромберг)", "epsE (Ейткен)"],
        [max(eps0, 1e-17), max(epsR, 1e-17), max(epsE, 1e-17)],
        color=["tab:blue", "tab:green", "tab:red"])
plt.yscale("log")
plt.title("Порівняння похибок при N0 = %d" % N0)
plt.ylabel("Абсолютна похибка")
plt.grid(True, axis="y", which="both")
show_figure(fig3, "fig3_compare.png")

# Рис.4. Похибки методів залежно від N0
fig4 = plt.figure("Рис.4 – Похибки методів залежно від N0", figsize=(8, 4.5))
tN = [r[0] for r in tab]
plt.semilogy(tN, [max(r[1], 1e-17) for r in tab], "o-", label="Сімпсон")
plt.semilogy(tN, [max(r[2], 1e-17) for r in tab], "s-", label="Рунге-Ромберг")
plt.semilogy(tN, [max(r[3], 1e-17) for r in tab], "^-", label="Ейткен")
plt.title("Похибки методів залежно від N0")
plt.xlabel("N0")
plt.ylabel("Абсолютна похибка")
plt.legend()
plt.grid(True, which="both")
show_figure(fig4, "fig4_methods_vs_N0.png")

# Рис.5. Адаптивний алгоритм: точність і кількість обчислень залежно від delta
fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.loglog(deltas, np.maximum(eps_ad, 1e-17), "o-")
ax1.set_title("Точність eps від параметра delta")
ax1.set_xlabel("delta")
ax1.set_ylabel("eps")
ax1.grid(True, which="both")
ax2.semilogx(deltas, evals_ad, "o-", color="tab:red")
ax2.set_title("Число обчислень f від параметра delta")
ax2.set_xlabel("delta")
ax2.set_ylabel("Кількість обчислень f(x)")
ax2.grid(True, which="both")
show_figure(fig5, "fig5_adaptive.png")