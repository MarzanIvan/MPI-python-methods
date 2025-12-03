from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot, sqrt, hstack
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()


def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part,
                              N, comm, comm_row, comm_col, rank):
    r_part = empty(N_part, dtype=float64)
    p_part = empty(N_part, dtype=float64)
    q_part = empty(N_part, dtype=float64)

    if rank in range(num_col):
        ScalP = array(0, dtype=float64)
        ScalP_temp = empty(1, dtype=float64)

    s = 1

    p_part = zeros(N_part, dtype=float64)

    while s <= N:

        if s == 1:
            comm_col.Bcast([x_part, N_part, MPI.DOUBLE], root=0)
            Ax_part_temp = dot(A_part, x_part)
            Ax_part = empty(M_part, dtype=float64)
            comm_row.Reduce([Ax_part_temp, M_part, MPI.DOUBLE],
                            [Ax_part, M_part, MPI.DOUBLE],
                            op=MPI.SUM, root=0)
            if rank in range(0, numprocs, num_col):
                b_part = Ax_part - b_part
            comm_row.Bcast([b_part, M_part, MPI.DOUBLE], root=0)
            r_part_temp = dot(A_part.T, b_part)
            comm_col.Reduce([r_part_temp, N_part, MPI.DOUBLE],
                            [r_part, N_part, MPI.DOUBLE],
                            op=MPI.SUM, root=0)
        else:
            if rank in range(num_col):
                ScalP_temp[0] = dot(p_part, q_part)
                comm_row.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                                   [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
                r_part = r_part - q_part / ScalP

        if rank in range(num_col):
            ScalP_temp[0] = dot(r_part, r_part)
            comm_row.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                               [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            p_part = p_part + r_part / ScalP

        comm_col.Bcast([p_part, N_part, MPI.DOUBLE], root=0)
        Ap_part_temp = dot(A_part, p_part)
        Ap_part = empty(M_part, dtype=float64)
        comm_row.Allreduce([Ap_part_temp, M_part, MPI.DOUBLE],
                           [Ap_part, M_part, MPI.DOUBLE], op=MPI.SUM)
        q_part_temp = dot(A_part.T, Ap_part)
        comm_col.Reduce([q_part_temp, N_part, MPI.DOUBLE],
                        [q_part, N_part, MPI.DOUBLE],
                        op=MPI.SUM, root=0)

        if rank in range(num_col):
            ScalP_temp[0] = dot(p_part, q_part)
            comm_row.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
                               [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            x_part = x_part - p_part / ScalP

        s = s + 1

    return x_part


if rank == 0:
    f1 = open('mp5/in.dat', 'r')
    N = array(int32(f1.readline()))
    M = array(int32(f1.readline()))
    f1.close()
else:
    N = array(0, dtype=int32)

comm.Bcast([N, 1, MPI.INT], root=0)

num_col = num_row = int32(sqrt(numprocs))


def auxiliary_arrays_determination(M, num):
    ave, res = divmod(M, num)
    rcounts = empty(num, dtype=int32)
    displs = empty(num, dtype=int32)
    for k in range(0, num):
        if k < res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        if k == 0:
            displs[k] = 0
        else:
            displs[k] = displs[k - 1] + rcounts[k - 1]
    return rcounts, displs


if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
else:
    rcounts_M = None
    displs_M = None
    rcounts_N = None
    displs_N = None

M_part = array(0, dtype=int32);
N_part = array(0, dtype=int32)

comm_col = comm.Split(rank % num_col, rank)
comm_row = comm.Split(rank // num_col, rank)

if rank in range(num_col):
    comm_row.Scatter([rcounts_N, 1, MPI.INT],
                     [N_part, 1, MPI.INT], root=0)
if rank in range(0, numprocs, num_col):
    comm_col.Scatter([rcounts_M, 1, MPI.INT],
                     [M_part, 1, MPI.INT], root=0)

comm_col.Bcast([N_part, 1, MPI.INT], root=0)
comm_row.Bcast([M_part, 1, MPI.INT], root=0)

A_part = empty((M_part, N_part), dtype=float64)

group = comm.Get_group()

if rank == 0:
    f2 = open('mp5/AData.dat', 'r')
    for m in range(num_row):
        a_temp = empty(rcounts_M[m] * N, dtype=float64)
        for j in range(rcounts_M[m]):
            for n in range(num_col):
                for i in range(rcounts_N[n]):
                    a_temp[rcounts_M[m] * displs_N[n] + j * rcounts_N[n] + i] = float64(f2.readline())
        if m == 0:
            comm_row.Scatterv([a_temp, rcounts_M[m] * rcounts_N, rcounts_M[m] * displs_N, MPI.DOUBLE],
                              [A_part, M_part * N_part, MPI.DOUBLE], root=0)
        else:
            group_temp = group.Range_incl([(0, 0, 1), (m * num_col, (m + 1) * num_col - 1, 1)])
            comm_temp = comm.Create(group_temp)
            rcounts_N_temp = hstack((array(0, dtype=int32), rcounts_N))
            displs_N_temp = hstack((array(0, dtype=int32), displs_N))
            comm_temp.Scatterv([a_temp, rcounts_M[m] * rcounts_N_temp, rcounts_M[m] * displs_N_temp, MPI.DOUBLE],
                               [empty(0, dtype=float64), 0, MPI.DOUBLE], root=0)
            group_temp.Free()
            comm_temp.Free()
    f2.close()
else:
    if rank in range(num_col):
        comm_row.Scatterv([None, None, None, None],
                          [A_part, M_part * N_part, MPI.DOUBLE], root=0)
    for m in range(1, num_row):
        group_temp = group.Range_incl([(0, 0, 1), (m * num_col, (m + 1) * num_col - 1, 1)])
        comm_temp = comm.Create(group_temp)
        if rank in range(m * num_col, (m + 1) * num_col):
            comm_temp.Scatterv([None, None, None, None],
                               [A_part, M_part * N_part, MPI.DOUBLE], root=0)
            comm_temp.Free()
        group_temp.Free()

if rank == 0:
    b = empty(M, dtype=float64)
    f3 = open('mp5/bData.dat', 'r')
    for j in range(M):
        b[j] = float64(f3.readline())
    f3.close()
else:
    b = None

b_part = empty(M_part, dtype=float64)

if rank in range(0, numprocs, num_col):
    comm_col.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
                      [b_part, M_part, MPI.DOUBLE], root=0)

if rank == 0:
    x = zeros(N, dtype=float64)
else:
    x = None

x_part = empty(N_part, dtype=float64)

if rank in range(num_col):
    comm_row.Scatterv([x, rcounts_N, displs_N, MPI.DOUBLE],
                      [x_part, N_part, MPI.DOUBLE], root=0)

x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part,
                                   N, comm, comm_row, comm_col, rank)

if rank in range(num_col):
    comm_row.Gatherv([x_part, N_part, MPI.DOUBLE],
                     [x, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    style.use('dark_background')
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i')
    ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, '-y', lw=3)
    show()
