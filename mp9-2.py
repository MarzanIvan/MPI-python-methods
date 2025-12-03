from mpi4py import MPI
from numpy import empty, array, int32, float64, linspace, sin, pi
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()

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


def parallel_tridiagonal_matrix_algorithm(a_part, b_part, c_part, d_part):
    N_part = len(d_part)

    for n in range(1, N_part):
        coef = a_part[n] / b_part[n - 1]
        a_part[n] = -coef * a_part[n - 1]
        b_part[n] = b_part[n] - coef * c_part[n - 1]
        d_part[n] = d_part[n] - coef * d_part[n - 1]

    for n in range(N_part - 3, -1, -1):
        coef = c_part[n] / b_part[n + 1]
        c_part[n] = -coef * c_part[n + 1]
        a_part[n] = a_part[n] - coef * a_part[n + 1]
        d_part[n] = d_part[n] - coef * d_part[n + 1]

    if rank_cart > 0:
        temp_array_send = array([a_part[0], b_part[0],
                                 c_part[0], d_part[0]], dtype=float64)
    if rank_cart < numprocs - 1:
        temp_array_recv = empty(4, dtype=float64)

    if rank_cart == 0:
        comm_cart.Recv([temp_array_recv, 4, MPI.DOUBLE],
                       source=1, tag=0, status=None)
    if rank_cart in range(1, numprocs - 1):
        comm_cart.Sendrecv(sendbuf=[temp_array_send, 4, MPI.DOUBLE],
                           dest=rank_cart - 1, sendtag=0,
                           recvbuf=[temp_array_recv, 4, MPI.DOUBLE],
                           source=rank_cart + 1, recvtag=MPI.ANY_TAG, status=None)
    if rank_cart == numprocs - 1:
        comm_cart.Send([temp_array_send, 4, MPI.DOUBLE],
                       dest=numprocs - 2, tag=0)

    if rank_cart < numprocs - 1:
        coef = c_part[N_part - 1] / temp_array_recv[1]
        b_part[N_part - 1] = b_part[N_part - 1] - coef * temp_array_recv[0]
        c_part[N_part - 1] = - coef * temp_array_recv[2]
        d_part[N_part - 1] = d_part[N_part - 1] - coef * temp_array_recv[3]

    temp_array_send = array([a_part[N_part - 1], b_part[N_part - 1],
                             c_part[N_part - 1], d_part[N_part - 1]],
                            dtype=float64)

    if rank_cart == 0:
        A_extended = empty((numprocs, 4), dtype=float64)
    else:
        A_extended = None

    comm_cart.Gather([temp_array_send, 4, MPI.DOUBLE],
                     [A_extended, 4, MPI.DOUBLE], root=0)

    if rank_cart == 0:
        x_temp = consecutive_tridiagonal_matrix_algorithm(
            A_extended[:, 0], A_extended[:, 1], A_extended[:, 2], A_extended[:, 3])
    else:
        x_temp = None

    if rank_cart == 0:
        rcounts_temp = empty(numprocs, dtype=int32)
        displs_temp = empty(numprocs, dtype=int32)
        rcounts_temp[0] = 1
        displs_temp[0] = 0
        for k in range(1, numprocs):
            rcounts_temp[k] = 2
            displs_temp[k] = k - 1
    else:
        rcounts_temp = None;
        displs_temp = None

    if rank_cart == 0:
        x_part_last = empty(1, dtype=float64)
        comm_cart.Scatterv([x_temp, rcounts_temp, displs_temp, MPI.DOUBLE],
                           [x_part_last, 1, MPI.DOUBLE], root=0)
    else:
        x_part_last = empty(2, dtype=float64)
        comm_cart.Scatterv([x_temp, rcounts_temp, displs_temp, MPI.DOUBLE],
                           [x_part_last, 2, MPI.DOUBLE], root=0)

    x_part = empty(N_part, dtype=float64)

    if rank_cart == 0:
        for n in range(N_part - 1):
            x_part[n] = (d_part[n] - c_part[n] * x_part_last[0]) / b_part[n]
        x_part[N_part - 1] = x_part_last[0]
    else:
        for n in range(N_part - 1):
            x_part[n] = (d_part[n] - a_part[n] * x_part_last[0]
                         - c_part[n] * x_part_last[1]) / b_part[n]
        x_part[N_part - 1] = x_part_last[1]

    return x_part


def f_part(y, t, h, N_part_aux, u_left, u_right, eps):
    N_part = N_part_aux - 2
    f_part = empty(N_part, dtype=float64)
    if rank_cart == 0:
        f_part[0] = eps * (y[2] - 2 * y[1] + u_left(t)) / h ** 2 + \
                    y[1] * (y[2] - u_left(t)) / (2 * h) + y[1] ** 3
        for n in range(2, N_part_aux - 1):
            f_part[n - 1] = eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2 + \
                            y[n] * (y[n + 1] - y[n - 1]) / (2 * h) + y[n] ** 3
    if rank_cart in range(1, numprocs - 1):
        for n in range(1, N_part_aux - 1):
            f_part[n - 1] = eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2 + \
                            y[n] * (y[n + 1] - y[n - 1]) / (2 * h) + y[n] ** 3
    if rank_cart == numprocs - 1:
        for n in range(1, N_part_aux - 2):
            f_part[n - 1] = eps * (y[n + 1] - 2 * y[n] + y[n - 1]) / h ** 2 + \
                            y[n] * (y[n + 1] - y[n - 1]) / (2 * h) + y[n] ** 3
        f_part[N_part - 1] = eps * (u_right(t) - 2 * y[N_part_aux - 2] + y[N_part_aux - 3]) / h ** 2 + \
                             y[N_part_aux - 2] * (u_right(t) - y[N_part_aux - 3]) / (2 * h) + y[N_part_aux - 2] ** 3
    return f_part


def diagonals_preparation(y, t, h, N_part_aux, u_left, u_right, eps, tau, alpha):
    N_part = N_part_aux - 2
    a_part = empty(N_part, dtype=float64)
    b_part = empty(N_part, dtype=float64)
    c_part = empty(N_part, dtype=float64)

    if rank_cart == 0:
        b_part[0] = 1. - alpha * tau * (-2 * eps / h ** 2 + (y[2] - u_left(t)) / (2 * h) + 3 * y[1] ** 2)
        c_part[0] = - alpha * tau * (eps / h ** 2 + y[1] / (2 * h))
        for n in range(2, N_part_aux - 1):
            a_part[n - 1] = - alpha * tau * (eps / h ** 2 - y[n] / (2 * h))
            b_part[n - 1] = 1. - alpha * tau * (-2 * eps / h ** 2 + (y[n + 1] - y[n - 1]) / (2 * h) + 3 * y[n] ** 2)
            c_part[n - 1] = - alpha * tau * (eps / h ** 2 + y[n] / (2 * h))
    if rank_cart in range(1, numprocs - 1):
        for n in range(1, N_part_aux - 1):
            a_part[n - 1] = - alpha * tau * (eps / h ** 2 - y[n] / (2 * h))
            b_part[n - 1] = 1. - alpha * tau * (-2 * eps / h ** 2 + (y[n + 1] - y[n - 1]) / (2 * h) + 3 * y[n] ** 2)
            c_part[n - 1] = - alpha * tau * (eps / h ** 2 + y[n] / (2 * h))
    if rank_cart == numprocs - 1:
        for n in range(1, N_part_aux - 2):
            a_part[n - 1] = - alpha * tau * (eps / h ** 2 - y[n] / (2 * h))
            b_part[n - 1] = 1. - alpha * tau * (-2 * eps / h ** 2 + (y[n + 1] - y[n - 1]) / (2 * h) + 3 * y[n] ** 2)
            c_part[n - 1] = - alpha * tau * (eps / h ** 2 + y[n] / (2 * h))
        a_part[N_part - 1] = - alpha * tau * (eps / h ** 2 - y[N_part_aux - 2] / (2 * h))
        b_part[N_part - 1] = 1. - alpha * tau * (-2 * eps / h ** 2 + (u_right(t) - y[N_part_aux - 3]) / (2 * h) + \
                                                 3 * y[N_part_aux - 2] ** 2)

    return a_part, b_part, c_part


if rank_cart == 0:
    start_time = MPI.Wtime()

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
    rcounts_aux = empty(numprocs, dtype=int32)
    displs_aux = empty(numprocs, dtype=int32)
    rcounts_aux[0] = rcounts[0] + 1
    displs_aux[0] = 0
    for k in range(1, numprocs - 1):
        rcounts_aux[k] = rcounts[k] + 2
        displs_aux[k] = displs[k] - 1
    rcounts_aux[numprocs - 1] = rcounts[numprocs - 1] + 1
    displs_aux[numprocs - 1] = displs[numprocs - 1] - 1
else:
    rcounts_aux = None;
    displs_aux = None

N_part_aux = array(0, dtype=int32);
displ_aux = array(0, dtype=int32)

comm_cart.Scatter([rcounts_aux, 1, MPI.INT], [N_part_aux, 1, MPI.INT], root=0)
comm_cart.Scatter([displs_aux, 1, MPI.INT], [displ_aux, 1, MPI.INT], root=0)

u_part_aux = empty((M + 1, N_part_aux), dtype=float64)

for n in range(N_part_aux):
    u_part_aux[0, n] = u_init(x[displ_aux + n])

y_part = u_part_aux[0, 1:N_part_aux - 1]

for m in range(M):

    codiagonal_down_part, diagonal_part, codiagonal_up_part = diagonals_preparation(
        u_part_aux[m], t[m], h, N_part_aux, u_left, u_right, eps, tau, alpha)
    w_1_part = parallel_tridiagonal_matrix_algorithm(
        codiagonal_down_part, diagonal_part, codiagonal_up_part,
        f_part(u_part_aux[m], t[m] + tau / 2, h, N_part_aux, u_left, u_right, eps))
    y_part = y_part + tau * w_1_part.real

    u_part_aux[m + 1, 1:N_part_aux - 1] = y_part
    if rank_cart == 0:
        u_part_aux[m + 1, 0] = u_left(t[m + 1])
    if rank_cart == numprocs - 1:
        u_part_aux[m + 1, N_part_aux - 1] = u_right(t[m + 1])

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
