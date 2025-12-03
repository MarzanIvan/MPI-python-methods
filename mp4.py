from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# TIMER: начало общего измерения
t_start = MPI.Wtime()


def conjugate_gradient_method(A_part, b_part, x_part,
                              N, N_part, rcounts_N, displs_N):

    # TIMER: начало метода
    t_cg_start = MPI.Wtime()

    x = empty(N, dtype=float64)
    p = empty(N, dtype=float64)
    r_part = empty(N_part, dtype=float64)
    p_part = empty(N_part, dtype=float64)
    q_part = empty(N_part, dtype=float64)

    ScalP = array(0, dtype=float64)
    ScalP_temp = empty(1, dtype=float64)

    s = 1
    p_part = 0.

    while s <= N:
        if s == 1:
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE],
                            [x, rcounts_N, displs_N, MPI.DOUBLE])
            r_temp = dot(A_part.T, dot(A_part, x) - b_part)
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE],
                                [r_part, N_part, MPI.DOUBLE],
                                recvcounts=rcounts_N, op=MPI.SUM)
        else:
            ScalP_temp[0] = dot(p_part, q_part)
            comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                           [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            r_part = r_part - q_part / ScalP

        ScalP_temp[0] = dot(r_part, r_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        p_part = p_part + r_part / ScalP

        comm.Allgatherv([p_part, N_part, MPI.DOUBLE],
                        [p, rcounts_N, displs_N, MPI.DOUBLE])
        q_temp = dot(A_part.T, dot(A_part, p))
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE],
                            [q_part, N_part, MPI.DOUBLE],
                            recvcounts=rcounts_N, op=MPI.SUM)

        ScalP_temp[0] = dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        x_part = x_part - p_part / ScalP

        s = s + 1

    # TIMER: конец метода
    t_cg_end = MPI.Wtime()
    if rank == 0:
        print(f"[TIMER] Conjugate Gradient time: {t_cg_end - t_cg_start:.6f} sec")

    return x_part


# Чтение входных данных (только процесс 0)
if rank == 0:
    t_io_start = MPI.Wtime()  # TIMER
    f1 = open('mp3/in.dat', 'r')
    N = array(int32(f1.readline()))
    M = array(int32(f1.readline()))
    f1.close()
    t_io_end = MPI.Wtime()  # TIMER
    print(f"[TIMER] Input reading time: {t_io_end - t_io_start:.6f} sec")
else:
    N = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)


def auxiliary_arrays_determination(M, numprocs):
    ave, res = divmod(M, numprocs - 1)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    rcounts[0] = 0
    displs[0] = 0
    for k in range(1, numprocs):
        if k < 1 + res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k - 1] + rcounts[k - 1]
    return rcounts, displs


if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, numprocs)
else:
    rcounts_M = None
    displs_M = None
    rcounts_N = empty(numprocs, dtype=int32)
    displs_N = empty(numprocs, dtype=int32)

M_part = array(0, dtype=int32)
N_part = array(0, dtype=int32)
comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)
comm.Bcast([rcounts_N, numprocs, MPI.INT], root=0)
comm.Bcast([displs_N, numprocs, MPI.INT], root=0)

# Распределение матрицы A
if rank == 0:
    f2 = open('mp3/AData.dat', 'r')
    for k in range(1, numprocs):
        A_part = empty((rcounts_M[k], N), dtype=float64)
        for j in range(rcounts_M[k]):
            for i in range(N):
                A_part[j, i] = float64(f2.readline())
        comm.Send([A_part, rcounts_M[k] * N, MPI.DOUBLE], dest=k, tag=0)
    f2.close()
    A_part = empty((M_part, N), dtype=float64)
else:
    A_part = empty((M_part, N), dtype=float64)
    comm.Recv([A_part, M_part * N, MPI.DOUBLE], source=0, tag=0, status=None)

# Распределение вектора b
if rank == 0:
    b = empty(M, dtype=float64)
    f3 = open('mp3/bData.dat', 'r')
    for j in range(M):
        b[j] = float64(f3.readline())
    f3.close()
else:
    b = None

b_part = empty(M_part, dtype=float64)
comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
              [b_part, M_part, MPI.DOUBLE], root=0)

if rank == 0:
    x = zeros(N, dtype=float64)
else:
    x = None

x_part = empty(rcounts_N[rank], dtype=float64)
comm.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
              [x_part, rcounts_N[rank], MPI.DOUBLE], root=0)

# TIMER: запуск метода
x_part = conjugate_gradient_method(A_part, b_part, x_part,
                                   N, rcounts_N[rank], rcounts_N, displs_N)

comm.Gatherv([x_part, rcounts_N[rank], MPI.DOUBLE],
             [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

# TIMER: конец общего времени
t_end = MPI.Wtime()
if rank == 0:
    print(f"[TIMER] Total execution time: {t_end - t_start:.6f} sec")

if rank == 0:
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i')
    ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, color='blue', linewidth=2)
    show()
