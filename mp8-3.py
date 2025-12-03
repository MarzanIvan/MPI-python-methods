from mpi4py import MPI
from numpy import empty, array, int32, float64, linspace, sin, pi
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

comm_cart = comm.Create_cart(dims=[numprocs], periods=[False], reorder=True)
rank_cart = comm_cart.Get_rank()


def u_init(x):
    u_init = sin(3 * pi * (x - 1 / 6))
    return u_init


def u_left(t):
    u_left = -1.
    return u_left


def u_right(t):
    u_right = 1.
    return u_right


if rank_cart == 0:
    start_time = MPI.Wtime()

a = 0.;
b = 1.
t_0 = 0.;
T = 6.0
eps = 10 ** (-1.5)

N = 200;
M = 20000

h = (b - a) / N;
x = linspace(a, b, N + 1)
tau = (T - t_0) / M;
t = linspace(t_0, T, M + 1)

if rank_cart == 0:
    ave, res = divmod(N + 1, numprocs)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    for k in range(0, numprocs):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k == 0:
            displs[k] = 0
        else:
            displs[k] = displs[k - 1] + rcounts[k - 1]
else:
    rcounts = None;
    displs = None

N_part = array(0, dtype=int32)

comm_cart.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)

if rank_cart == 0:
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
    rcounts_from_0 = None;
    displs_from_0 = None

N_part_aux = array(0, dtype=int32);
displ_aux = array(0, dtype=int32)

comm_cart.Scatter([rcounts_from_0, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)
comm_cart.Scatter([displs_from_0, 1, MPI.INT], [displ_aux, 1, MPI.INT], root=0)

u_part_aux = empty((M + 1, N_part_aux), dtype=float64)

for n in range(N_part_aux):
    u_part_aux[0, n] = u_init(x[displ_aux + n])
if rank_cart == 0:
    for m in range(1, M + 1):
        u_part_aux[m, 0] = u_left(t[m])
if rank_cart == numprocs - 1:
    for m in range(1, M + 1):
        u_part_aux[m, N_part_aux - 1] = u_right(t[m])

for m in range(M):

    for n in range(1, N_part_aux - 1):
        u_part_aux[m + 1, n] = u_part_aux[m, n] + \
                               eps * tau / h ** 2 * (
                                           u_part_aux[m, n + 1] - 2 * u_part_aux[m, n] + u_part_aux[m, n - 1]) + \
                               tau / (2 * h) * u_part_aux[m, n] * (u_part_aux[m, n + 1] - u_part_aux[m, n - 1]) + \
                               tau * u_part_aux[m, n] ** 3

    if rank_cart == 0:
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m + 1, N_part_aux - 2], 1, MPI.DOUBLE],
                           dest=1, sendtag=0,
                           recvbuf=[u_part_aux[m + 1, N_part_aux - 1:], 1, MPI.DOUBLE],
                           source=1, recvtag=MPI.ANY_TAG, status=None)
    if rank_cart in range(1, numprocs - 1):
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m + 1, 1], 1, MPI.DOUBLE],
                           dest=rank_cart - 1, sendtag=0,
                           recvbuf=[u_part_aux[m + 1, 0:], 1, MPI.DOUBLE],
                           source=rank_cart - 1, recvtag=MPI.ANY_TAG, status=None)
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m + 1, N_part_aux - 2], 1, MPI.DOUBLE],
                           dest=rank_cart + 1, sendtag=0,
                           recvbuf=[u_part_aux[m + 1, N_part_aux - 1:], 1, MPI.DOUBLE],
                           source=rank_cart + 1, recvtag=MPI.ANY_TAG, status=None)
    if rank_cart == numprocs - 1:
        comm_cart.Sendrecv(sendbuf=[u_part_aux[m + 1, 1], 1, MPI.DOUBLE],
                           dest=numprocs - 2, sendtag=0,
                           recvbuf=[u_part_aux[m + 1, 0:], 1, MPI.DOUBLE],
                           source=numprocs - 2, recvtag=MPI.ANY_TAG, status=None)

if rank_cart == 0:
    u_T = empty(N + 1, dtype=float64)
else:
    u_T = None

if rank_cart == 0:
    comm_cart.Gatherv([u_part_aux[M, 0:N_part_aux - 1], N_part, MPI.DOUBLE],
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)
if rank_cart in range(1, numprocs - 1):
    comm_cart.Gatherv([u_part_aux[M, 1:N_part_aux - 1], N_part, MPI.DOUBLE],
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)
if rank_cart == numprocs - 1:
    comm_cart.Gatherv([u_part_aux[M, 1:N_part_aux], N_part, MPI.DOUBLE],
                      [u_T, rcounts, displs, MPI.DOUBLE], root=0)

if rank_cart == 0:
    end_time = MPI.Wtime()
    print('N={}, M={}'.format(N, M))
    print('Number of MPI process is {}'.format(numprocs))
    print('Elapsed time is {:.4f} sec.'.format(end_time - start_time))

    style.use('dark_background')
    fig = figure()
    ax = axes(xlim=(a, b), ylim=(-2.0, 2.0))
    ax.set_xlabel('x');
    ax.set_ylabel('u')
    ax.plot(x, u_T, color='y', ls='-', lw=2)
    show()
