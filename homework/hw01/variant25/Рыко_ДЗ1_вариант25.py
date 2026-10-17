"""ДЗ1, вариант 25.

Источники: «Задание 1.pdf», с. 10; «Семинар 6 осень_ Сплайны.pdf», с. 1–3;
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BarycentricInterpolator.html
https://mpmath.org/doc/current/ ; https://docs.python.org/3/library/decimal.html
Запуск: python -X utf8 Рыко_ДЗ1_вариант25.py
Зависимости: numpy scipy matplotlib mpmath. Время: несколько минут.
N — число узлов, номинальная степень полинома — N-1.
"""
from pathlib import Path
from decimal import Decimal, getcontext
import csv
import json
import platform
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BarycentricInterpolator
from scipy.optimize import minimize_scalar
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "результаты"
OUT.mkdir(exist_ok=True)
getcontext().prec = 90
mp.mp.dps = 110
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def dec(x):
    # 105 значащих цифр перед округлением арифметики Decimal до 90.
    return Decimal(mp.nstr(x, 105))


def nodes(n, grid, a, b):
    if grid == "uniform":
        return [a + (b-a)*j/(n-1) for j in range(n)]
    return sorted([(a+b)/2 + (b-a)/2*mp.cos((2*j+1)*mp.pi/(2*n))
                   for j in range(n)])


def newton_coefficients(x, y):
    # Обратный порядок обновления сохраняет предыдущий столбец разностей.
    c = list(y)
    for k in range(1, len(x)):
        for j in range(len(x)-1, k-1, -1):
            c[j] = (c[j]-c[j-1])/(x[j]-x[j-k])
    return c


def polynomial(t, x, c):
    # Схема Горнера для формы Ньютона, без системы Вандермонда.
    value = c[-1]
    for j in range(len(c)-2, -1, -1):
        value = c[j] + (t-x[j])*value
    return value


def numerical_maximum(x, errors, error_at):
    # Сетка даёт кандидатов; уточняем пять наибольших локальных максимумов.
    candidates = np.where((errors[1:-1] >= errors[:-2]) &
                          (errors[1:-1] >= errors[2:]))[0] + 1
    candidates = sorted(candidates, key=lambda j: errors[j], reverse=True)[:5]
    best = (float(np.max(errors)), float(x[np.argmax(errors)]))
    for j in candidates:
        r = minimize_scalar(lambda t: -error_at(t),
                            bounds=(x[j-1], x[j+1]), method="bounded",
                            options={"xatol": 1e-14})
        best = max(best, (error_at(r.x), float(r.x)))
    return best  # Численная оценка, а не доказанный супремум.


def natural_extension(spline, x, y, t):
    # Вне крайних узлов — касательные прямые. Новых узлов не добавляем.
    t = np.asarray(t)
    v = spline(np.clip(t, x[0], x[-1]))
    v = np.where(t < x[0], y[0] + spline(x[0], 1)*(t-x[0]), v)
    return np.where(t > x[-1], y[-1] + spline(x[-1], 1)*(t-x[-1]), v)


def bounds(n, grid, x, a, b, name):
    length = b-a
    # Для cos модуль любой производной <= 1.
    # Для tanh положим t=tanh(x). Производные — полиномы Q_N(t):
    # Q_0=t, Q_(k+1)=(1-t*t)*Q_k'. При |t|<=1 их модуль
    # не больше суммы модулей коэффициентов. Все коэффициенты — целые.
    m_n = 1
    if name == "tanh":
        q = [0, 1]  # Коэффициенты при степенях t: от младшей к старшей.
        for _ in range(n):
            derivative = [j*q[j] for j in range(1, len(q))]
            q = derivative + [0, 0]
            for j, value in enumerate(derivative):
                q[j+2] -= value
        m_n = sum(abs(value) for value in q)
    derivative_factor = mp.mpf(m_n)/mp.factorial(n)
    omega = (length/(n-1))**n*mp.factorial(n-1)/4
    if grid == "chebyshev":
        omega = 2*(length/4)**n
    p_bound = derivative_factor*omega
    h = max(x[j+1]-x[j] for j in range(n-1))
    m2 = 1 if name == "cos" else 4/(3*mp.sqrt(3))
    left, right = x[0]-a, b-x[-1]
    s_bound = m2*max(h*h/2, left*(x[1]-x[0])+left*left/2,
                    right*(x[-1]-x[-2])+right*right/2)
    return float(p_bound), float(s_bound)


def independent_check(n, grid, a, b, f, peak, expected):
    # Независимая форма Лагранжа при 120 цифрах проверяет максимум Ньютона.
    with mp.workdps(120):
        aa, bb = (mp.mpf(0), 2*mp.pi) if a == 0 else (-mp.pi, mp.pi)
        x = nodes(n, grid, aa, bb)
        t = mp.mpf(str(peak))
        w = [1/mp.fprod(x[j]-x[k] for k in range(n) if j != k)
             for j in range(n)]
        p = mp.fsum(wj*f(xj)/(t-xj) for wj, xj in zip(w, x))
        p /= mp.fsum(wj/(t-xj) for wj, xj in zip(w, x))
        measured = float(abs(p-f(t)))
    relative = abs(measured-expected)/expected
    assert relative < 1e-9, (n, grid, relative)
    return relative


def main():
    rows, curves, checks = [], {}, []
    for name, f, ff, a, b in [("cos", mp.cos, np.cos, mp.mpf(0), 2*mp.pi),
                                ("tanh", mp.tanh, np.tanh, -mp.pi, mp.pi)]:
        # Объединяем равномерную и сгущённую к краям проверочные сетки.
        mesh = sorted(set([a+(b-a)*j/2000 for j in range(2001)] +
                          [(a+b)/2+(b-a)/2*mp.cos(mp.pi*j/2000)
                           for j in range(2001)]))
        xd, yd = [dec(t) for t in mesh], [dec(f(t)) for t in mesh]
        xf = np.array([float(t) for t in mesh])
        yf = np.array([float(f(t)) for t in mesh])
        for grid in ["uniform", "chebyshev"]:
            for n in range(20, 61):
                xm = nodes(n, grid, a, b)
                xx, yy = [dec(t) for t in xm], [dec(f(t)) for t in xm]
                c = newton_coefficients(xx, yy)
                assert max(abs(polynomial(t, xx, c)-v) for t, v in zip(xx, yy)) < Decimal('1e-65')
                pd = [polynomial(t, xx, c) for t in xd]
                pe = np.array([float(abs(v-y)) for v, y in zip(pd, yd)])

                def ep(t):
                    q = mp.mpf(str(t))
                    return float(abs(polynomial(dec(q), xx, c)-dec(f(q))))

                pmax, pwhere = numerical_maximum(xf, pe, ep)
                delta = independent_check(n, grid, a, b, f, pwhere, pmax)
                x = np.array([float(t) for t in xm])
                y = np.array([float(f(t)) for t in xm])
                s = CubicSpline(x, y, bc_type="natural")
                assert np.max(abs(s(x)-y)) < 1e-12
                assert np.max(abs(s(x[[0, -1]], 2))) < 1e-10
                sv = natural_extension(s, x, y, xf)
                se = abs(sv-yf)
                es = lambda t: float(abs(natural_extension(s, x, y, t)-ff(t)))
                smax, swhere = numerical_maximum(xf, se, es)
                bp, bs = bounds(n, grid, xm, a, b, name)
                assert pmax <= bp*(1+1e-9) and smax <= bs*(1+1e-9)
                # Обычная float64-реализация нужна только для сравнения округлений.
                p64 = BarycentricInterpolator(x, y, rng=0)(xf)
                row = dict(function=name, grid=grid, N=n, degree=n-1,
                           polynomial_error=pmax, spline_error=smax,
                           polynomial_bound=bp, spline_bound=bs,
                           float64_error=float(np.max(abs(p64-yf))),
                           x_polynomial_max=pwhere, x_spline_max=swhere,
                           independent_relative_difference=delta)
                rows.append(row)
                if n in [20, 21, 40, 41, 59, 60]:
                    # Вторая, более густая сетка проверяет поиск пиков.
                    dense = np.linspace(float(a), float(b), 8001)
                    dense_p = max(ep(t) for t in dense)
                    dense_s = float(np.max(abs(natural_extension(s, x, y, dense)-ff(dense))))
                    assert dense_p <= pmax*1.00001 and dense_s <= smax*1.00001
                    checks.append(dict(function=name, grid=grid, N=n,
                                       dense_polynomial_ratio=dense_p/pmax,
                                       dense_spline_ratio=dense_s/smax))
                if n in [20, 21, 60]:
                    curves[f"{name}_{grid}_{n}"] = dict(x=xf, f=yf,
                        p=np.array([float(v) for v in pd]), s=sv, pe=pe, se=se)
                print(name, grid, n, f"P={pmax:.5e} S={smax:.5e}", flush=True)
    with (OUT / "ошибки.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "проверки.json").write_text(json.dumps(dict(
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, mpmath=mp.__version__, matplotlib=matplotlib.__version__),
        decimal_digits=90, rows=len(rows), dense_checks=checks,
        largest_independent_relative_difference=max(r['independent_relative_difference'] for r in rows)
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "кривые.npz", **{f"{k}__{v}": arr
        for k, data in curves.items() for v, arr in data.items()})
    draw(rows, curves)


def draw(rows, curves):
    colors = {"uniform": "#B75030", "chebyshev": "#176C8C"}
    labels = {"uniform": "Равномерная", "chebyshev": "Нули Чебышёва"}
    for name in ["cos", "tanh"]:
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
        for j, grid in enumerate(colors):
            v = curves[f"{name}_{grid}_20"]
            axes[0,j].plot(v['x'],v['f'],color='black',lw=2,label='f(x)')
            axes[0,j].plot(v['x'],v['p'],'--',color=colors[grid],label='P₁₉(x)')
            axes[0,j].plot(v['x'],v['s'],':',color='#7D3C98',label='S(x)')
            axes[0,j].set_title(labels[grid]+", N=20")
            for n, style in [(20,'-'),(21,'--'),(60,':')]:
                u=curves[f"{name}_{grid}_{n}"]
                axes[1,j].semilogy(u['x'],np.maximum(u['pe'],1e-85),style,
                                  label=f'Полином, N={n}')
            axes[1,j].semilogy(v['x'],np.maximum(v['se'],1e-85),color='#7D3C98',label='Сплайн, N=20')
            axes[1,j].set_ylim(bottom=1e-76 if name=='cos' else 1e-16)
        for ax in axes.flat:
            ax.set_xlabel('x'); ax.grid(alpha=.25); ax.legend(fontsize=10)
        for ax in axes[0]: ax.set_ylabel('Значение функции')
        for ax in axes[1]: ax.set_ylabel('Абсолютная ошибка')
        fig.suptitle(f'{name}(x): интерполяция и распределение ошибок')
        fig.savefig(OUT / f'{name}_curves.png',dpi=190); plt.close(fig)
        fig, axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
        for grid in colors:
            rr=[r for r in rows if r['function']==name and r['grid']==grid]
            for ax,method in zip(axes,['polynomial','spline']):
                ns=[r['N'] for r in rr]
                ax.semilogy(ns,[r[method+'_error'] for r in rr],'-o',ms=3,
                            color=colors[grid],label=labels[grid]+': ошибка')
                # Грубая растущая граница для полинома tanh
                # дана в CSV и отчёте; здесь она скрыла бы полезные кривые.
                if not (name=='tanh' and method=='polynomial'):
                    ax.semilogy(ns,[r[method+'_bound'] for r in rr],'--',
                                color=colors[grid],label=labels[grid]+': оценка')
        for ax,title in zip(axes,['Полином','Естественный сплайн']):
            ax.set_title(title); ax.set_xlabel('N — число узлов')
            ax.set_ylabel('Максимальная ошибка / оценка')
            ax.grid(alpha=.25); ax.legend(fontsize=9)
        fig.suptitle(f'{name}(x): ошибка и априорная граница')
        fig.savefig(OUT/f'{name}_convergence.png',dpi=190); plt.close(fig)


if __name__ == "__main__":
    main()
