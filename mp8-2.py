from numpy import empty, linspace, sin, pi, clip, savez
import time


def u_init(x):
    return sin(3 * pi * (x - 1 / 6))


def u_left(t):
    return -1.


def u_right(t):
    return 1.


start_time = time.time()

a, b = 0., 1.
t_0, T = 0., 6.0
eps = 10 ** (-1.5)

N, M = 100, 1000

h = (b - a) / N
x = linspace(a, b, N + 1)
tau = (T - t_0) / M
t = linspace(t_0, T, M + 1)

u = empty((M + 1, N + 1))

for n in range(N + 1):
    u[0, n] = u_init(x[n])

for m in range(M):
    u[m + 1, 0] = u_left(t[m + 1])
    u[m + 1, N] = u_right(t[m + 1])

    # временный массив для новых значений
    u_new = empty(N - 1, dtype=float)

    for n in range(1, N):
        laplace = eps * tau / h ** 2 * (u[m, n + 1] - 2 * u[m, n] + u[m, n - 1])
        convection = tau / (2 * h) * u[m, n] * (u[m, n + 1] - u[m, n - 1])
        nonlinear = tau * u[m, n] ** 3
        u_new[n - 1] = clip(u[m, n] + laplace + convection + nonlinear, -1e6, 1e6)

    u[m + 1, 1:N] = u_new

end_time = time.time()

print(f'Elapsed time is {end_time - start_time:.4f} sec')

savez('results_of_calculations', x=x, u=u)
