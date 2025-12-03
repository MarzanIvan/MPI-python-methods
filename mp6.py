from mpi4py import MPI
import numpy as np
from math import sqrt
from matplotlib.pyplot import style, figure, axes, show

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

def auxiliary_arrays_determination(M, num):
    ave, res = divmod(M, num)
    rcounts = np.empty(num, dtype=np.int32)
    displs = np.empty(num, dtype=np.int32)
    s = 0
    for k in range(num):
        rcounts[k] = ave + (1 if k < res else 0)
        displs[k] = s
        s += int(rcounts[k])
    return rcounts, displs

def conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part,
                              N_global, comm_row, comm_col):
    # A_part shape (M_part, N_part)
    # b_part shape (M_part,)
    # x_part shape (N_part,)
    # comm_row groups processes across columns with same row index
    # comm_col groups processes across rows with same col index

    # подготовка рабочих массивов
    r_part = np.zeros(N_part, dtype=np.float64)
    p_part = np.zeros(N_part, dtype=np.float64)
    q_part = np.zeros(N_part, dtype=np.float64)

    ScalP = np.zeros(1, dtype=np.float64)
    ScalP_temp = np.zeros(1, dtype=np.float64)

    # начальное восстановление r = A^T (A x - b)
    # сначала локальные взносы Ax_local = A_part @ x_part (разные столбцы дают вклад в одни и те же строки)
    Ax_local = A_part.dot(x_part)   # shape (M_part,)
    Ax_full = np.zeros(M_part, dtype=np.float64)
    comm_row.Allreduce(Ax_local, Ax_full, op=MPI.SUM)  # суммируем по столбцам -> полный Ax для локальных строк на этой row

    b_local = b_part.copy()
    # r_rows = Ax_full - b_local, затем r_part = A_part.T @ r_rows, и суммирование по строкам (comm_col)
    r_rows = Ax_full - b_local
    r_part_temp = A_part.T.dot(r_rows)   # shape (N_part,)
    comm_col.Allreduce(r_part_temp, r_part, op=MPI.SUM)  # суммируем по строкам -> r_part (локальная часть по колонкам)

    p_part[:] = 0.0

    # основной цикл: не больше N_global итераций (можно поставить условие по невязке)
    for s in range(1, int(N_global) + 1):
        # compute alpha-related quantities
        ScalP_temp[0] = np.dot(r_part, r_part)
        comm_row.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)  # scalar = sum(r^2) over row
        # если масштаб почти ноль — выход
        if ScalP[0] == 0.0:
            break
        p_part = p_part + r_part / ScalP[0]

        # Ap_local = A_part @ p_part
        Ap_local = A_part.dot(p_part)   # shape (M_part,)
        Ap_full = np.zeros(M_part, dtype=np.float64)
        comm_row.Allreduce(Ap_local, Ap_full, op=MPI.SUM)

        q_part_temp = A_part.T.dot(Ap_full)
        comm_col.Allreduce(q_part_temp, q_part, op=MPI.SUM)

        ScalP_temp[0] = np.dot(p_part, q_part)
        comm_row.Allreduce(ScalP_temp, ScalP, op=MPI.SUM)
        if ScalP[0] == 0.0:
            break

        x_part = x_part - p_part / ScalP[0]
        r_part = r_part - q_part / ScalP[0]

    return x_part

# root читает N и M и распределяет базовые массивы rcounts/displs
if rank == 0:
    with open('mp6/in.dat', 'r') as f:
        N = int(f.readline().strip())
        M = int(f.readline().strip())
else:
    N = None
    M = None

N = comm.bcast(N, root=0)
M = comm.bcast(M, root=0)

# требуемое число процессов — квадрат
root_ok = True
num_row = int(sqrt(numprocs))
num_col = num_row
if num_row * num_col != numprocs:
    if rank == 0:
        print("ERROR: number of processes must be a perfect square (e.g. 4, 9, 16).")
    MPI.Finalize()
    raise SystemExit(1)

# вычисляем rcounts/displs для строк (разбиение M) и столбцов (разбиение N)
if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
else:
    rcounts_M = None; displs_M = None
    rcounts_N = None; displs_N = None

# каждый процесс может определить свою координату в сетке
row_idx = rank // num_col
col_idx = rank % num_col

# бродкастим rcounts/displs полностью всем (удобно и безопасно)
rcounts_M = comm.bcast(rcounts_M, root=0)
displs_M = comm.bcast(displs_M, root=0)
rcounts_N = comm.bcast(rcounts_N, root=0)
displs_N = comm.bcast(displs_N, root=0)

# определяем локальные размеры как python int
M_part = int(rcounts_M[row_idx])
N_part = int(rcounts_N[col_idx])

# создаём коммуникаторы-строку и -столбец
comm_row = comm.Split(row_idx, rank)
comm_col = comm.Split(col_idx, rank)

# создаём локальные буферы
A_part = np.empty((M_part, N_part), dtype=np.float64)
b_part = np.empty(M_part, dtype=np.float64)
x_part = np.empty(N_part, dtype=np.float64)

# root читает A полный и отправляет блоки каждому процессу (простая и надёжная логика)
if rank == 0:
    # читаем AData.dat как построчно (M строк по N чисел)
    A_full = np.empty((M, N), dtype=np.float64)
    with open('mp6/AData.dat', 'r') as fA:
        for i in range(M):
            for j in range(N):
                try:
                    A_full[i, j] = float(fA.readline().strip())
                except Exception:
                    A_full[i, j] = 0.0  # на случай короче файла
    # рассылаем блоки
    for proc in range(numprocs):
        proc_row = proc // num_col
        proc_col = proc % num_col
        r0 = int(displs_M[proc_row])
        r1 = r0 + int(rcounts_M[proc_row])
        c0 = int(displs_N[proc_col])
        c1 = c0 + int(rcounts_N[proc_col])
        block = A_full[r0:r1, c0:c1].copy()  # важен копируемый буфер
        if proc == 0:
            A_part[:, :] = block
        else:
            comm.Send([block, MPI.DOUBLE], dest=proc, tag=10+proc)
    # читаем b и рассылаем
    b_full = np.zeros(M, dtype=np.float64)
    with open('mp6/bData.dat', 'r') as fb:
        for i in range(M):
            try:
                b_full[i] = float(fb.readline().strip())
            except Exception:
                b_full[i] = 0.0
    for proc in range(numprocs):
        proc_row = proc // num_col
        r0 = int(displs_M[proc_row])
        r1 = r0 + int(rcounts_M[proc_row])
        block_b = b_full[r0:r1].copy()
        if proc == 0:
            b_part[:] = block_b
        else:
            comm.Send([block_b, MPI.DOUBLE], dest=proc, tag=200+proc)
    # начальное x (нулевой вектор)
    x_full = np.zeros(N, dtype=np.float64)
    for proc in range(numprocs):
        proc_col = proc % num_col
        c0 = int(displs_N[proc_col])
        c1 = c0 + int(rcounts_N[proc_col])
        block_x = x_full[c0:c1].copy()
        if proc == 0:
            x_part[:] = block_x
        else:
            comm.Send([block_x, MPI.DOUBLE], dest=proc, tag=300+proc)
else:
    # принимаем свой блок матрицы, b и x
    req = comm.Recv([A_part, MPI.DOUBLE], source=0, tag=10+rank)
    req = comm.Recv([b_part, MPI.DOUBLE], source=0, tag=200+rank)
    req = comm.Recv([x_part, MPI.DOUBLE], source=0, tag=300+rank)

# убедимся, что типы и размеры корректны (опционально, можно раскомментировать)
# print(f"rank {rank}: M_part={M_part}, N_part={N_part}")

# запускаем метод сопряжённых градиентов
x_part = conjugate_gradient_method(A_part, b_part, x_part, N_part, M_part, N, comm_row, comm_col)

# собираем x_parts на root (root собирает с каждого процесса соответствующую часть)
if rank == 0:
    x_full = np.empty(N, dtype=np.float64)
    # помещаем свои данные
    c0 = int(displs_N[0])
    c1 = c0 + int(rcounts_N[0])
    x_full[c0:c1] = x_part.copy()
    for proc in range(1, numprocs):
        proc_col = proc % num_col
        c0 = int(displs_N[proc_col])
        c1 = c0 + int(rcounts_N[proc_col])
        recvbuf = np.empty(c1 - c0, dtype=np.float64)
        comm.Recv([recvbuf, MPI.DOUBLE], source=proc, tag=400+proc)
        x_full[c0:c1] = recvbuf
else:
    comm.Send([x_part, MPI.DOUBLE], dest=0, tag=400+rank)

# root визуализирует результат
if rank == 0:
    style.use('dark_background')
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(np.min(x_full)-0.1, np.max(x_full)+0.1))
    ax.set_xlabel('i'); ax.set_ylabel('x[i]')
    ax.plot(np.arange(N), x_full, '-y', lw=2)
    show()
