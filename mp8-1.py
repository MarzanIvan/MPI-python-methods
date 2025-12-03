from mpi4py import MPI
from numpy import empty, array, int32, float64, linspace, sin, pi, hstack, clip, savez

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()


def u_init(x):
    return sin(3 * pi * (x - 1 / 6))


def u_left(t):
    return -1.


def u_right(t):
    return 1.


if rank == 0:
    start_time = MPI.Wtime()

a, b = 0., 1.
t_0, T = 0., 6.0
eps = 10 ** (-1.5)

N, M = 100, 1000

h = (b - a) / N
x = linspace(a, b, N + 1)
tau = (T - t_0) / M
t = linspace(t_0, T, M + 1)

if rank == 0:
    ave, res = divmod(N + 1, numprocs)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    for k in range(numprocs):
        rcounts[k] = ave + 1 if k < res else ave
        displs[k] = 0 if k == 0 else displs[k - 1] + rcounts[k - 1]
else:
    rcounts = None
    displs = None

N_part = array(0, dtype=int32)
comm.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

if rank == 0:
    rcounts_from_0 = empty(numprocs, dtype=int32)
    displs_from_0 = empty(numprocs, dtype=int32)
    rcounts_from_0[0] = rcounts[0] + 1
    displs_from_0[0] = 0
    for k in range(1, numprocs - 1):
        rcounts_from_0[k] = rcounts[k] + 2
        displs_from_0[k] = displs[k] - 1
    rcounts_from_0[numprocs - 1] = rcounts[numprocs - 1] + 1
    displs_from_0[numprocs - 1] = displs[numprocs - 1] - 1
else:
    rcounts_from_0 = None
    displs_from_0 = None

N_part_aux = array(0, dtype=int32)
comm.Scatter([rcounts_from_0, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)

if rank == 0:
    u = empty((M + 1, N + 1), dtype=float64)
    for n in range(N + 1):
        u[0, n] = u_init(x[n])
else:
    u = empty((M + 1, 0), dtype=float64)

u_part = empty(N_part, dtype=float64)
u_part_aux = empty(N_part_aux, dtype=float64)

for m in range(M):

    comm.Scatterv([u[m], rcounts_from_0, displs_from_0, MPI.DOUBLE],
                  [u_part_aux, N_part_aux, MPI.DOUBLE], root=0)

    # временный массив для вычисления новых значений
    u_part_new = empty(N_part, dtype=float64)

    for n in range(1, N_part_aux - 1):
        laplace = eps * tau / h ** 2 * (u_part_aux[n + 1] - 2 * u_part_aux[n] + u_part_aux[n - 1])
        convection = tau / (2 * h) * u_part_aux[n] * (u_part_aux[n + 1] - u_part_aux[n - 1])
        nonlinear = tau * (u_part_aux[n] ** 3)
        # ограничиваем значение, чтобы избежать overflow
        u_part_new[n - 1] = clip(u_part_aux[n] + laplace + convection + nonlinear, -1e6, 1e6)

    # граничные условия
    if rank == 0:
        u_part = hstack((array(u_left(t[m + 1]), dtype=float64), u_part_new[0:N_part - 1]))
    elif rank == numprocs - 1:
        u_part = hstack((u_part_new[0:N_part - 1], array(u_right(t[m + 1]), dtype=float64)))
    else:
        u_part = u_part_new.copy()

    comm.Gatherv([u_part, N_part, MPI.DOUBLE],
                 [u[m + 1], rcounts, displs, MPI.DOUBLE], root=0)

if rank == 0:
    end_time = MPI.Wtime()
    print(f'N={N}, M={M}')
    print(f'Number of MPI process is {numprocs}')
    print(f'Elapsed time is {end_time - start_time:.4f} sec.')

    savez('results_of_calculations', x=x, u=u)
