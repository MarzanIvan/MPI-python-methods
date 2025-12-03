from numpy import empty, linspace, sin, pi, float64
import time


def u_init(x):
    u_init = sin(3 * pi * (x - 1 / 6))
    return u_init


def u_left(t):
    u_left = -1.
    return u_left


def u_right(t):
    u_right = 1.
    return u_right


def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    N = len(d)

    x = empty(N, dtype=float64)

    for n in range(1, N):
        coef = a[n] / b[n - 1]
        b[n] = b[n] - coef * c[n - 1]
        d[n] = d[n] - coef * d[n - 1]

    x[N - 1] = d[N - 1] / b[N - 1]

    for n in range(N - 2, -1, -1):
        x[n] = (d[n] - c[n] * x[n + 1]) / b[n]

    return x


def f(y, t, h, N, u_left, u_right, eps):
    f = empty(N - 1, dtype=float64)
    f[0] = eps * (y[1] - 2 * y[0] + u_left(t)) / h ** 2 + y[0] * (y[1] - u_left(t)) / (2 * h) + y[0] ** 3
    for n in range(1, N - 2):
        f[n] = eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2 + y[n] * (y[n + 1] - y[n - 1]) / (2 * h) + y[n] ** 3
    f[N - 2] = eps * (u_right(t) - 2 * y[N - 2] + y[N - 3]) / h ** 2 + y[N - 2] * (u_right(t) - y[N - 3]) / (2 * h) + y[
        N - 2] ** 3
    return f


def diagonals_preparation(y, t, h, N, u_left, u_right, eps, tau, alpha):
    a = empty(N - 1, dtype=float64)
    b = empty(N - 1, dtype=float64)
    c = empty(N - 1, dtype=float64)

    b[0] = 1. - alpha * tau * (-2 * eps / h ** 2 + (y[1] - u_left(t)) / (2 * h) + 3 * y[0] ** 2)
    c[0] = - alpha * tau * (eps / h ** 2 + y[0] / (2 * h))
    for n in range(1, N - 2):
        a[n] = - alpha * tau * (eps / h ** 2 - y[n] / (2 * h))
        b[n] = 1. - alpha * tau * (-2 * eps / h ** 2 + (y[n + 1] - y[n - 1]) / (2 * h) + 3 * y[n] ** 2)
        c[n] = - alpha * tau * (eps / h ** 2 + y[n] / (2 * h))
    a[N - 2] = - alpha * tau * (eps / h ** 2 - y[N - 2] / (2 * h))
    b[N - 2] = 1. - alpha * tau * (-2 * eps / h ** 2 + (u_right(t) - y[N - 3]) / (2 * h) + 3 * y[N - 2] ** 2)

    return a, b, c


start_time = time.time()

a = 0.;
b = 1.
t_0 = 0.;
T = 2.0
eps = 10 ** (-1.5)

N = 200;
M = 300;
alpha = 0.5

h = (b - a) / N;
x = linspace(a, b, N + 1)
tau = (T - t_0) / M;
t = linspace(t_0, T, M + 1)

u = empty((M + 1, N + 1))

for n in range(N + 1):
    u[0, n] = u_init(x[n])

y = u_init(x[1:N])

for m in range(M):
    codiagonal_down, diagonal, codiagonal_up = diagonals_preparation(
        y, t[m], h, N, u_left, u_right, eps, tau, alpha)
    w_1 = consecutive_tridiagonal_matrix_algorithm(
        codiagonal_down, diagonal, codiagonal_up,
        f(y, t[m] + tau / 2, h, N, u_left, u_right, eps))
    y = y + tau * w_1.real

    u[m + 1, 0] = u_left(t[m + 1])
    u[m + 1, 1:N] = y
    u[m + 1, N] = u_right(t[m + 1])

end_time = time.time()

print('Elapsed time is {:.4f} sec'.format(end_time - start_time))

from numpy import savez

savez('results_of_calculations', x=x, t=t, u=u)
