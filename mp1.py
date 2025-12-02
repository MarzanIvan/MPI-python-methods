from mpi4py import MPI
from numpy import empty, array, int32, float64, dot

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

# Читаем N, M только на 0-м процессе
if rank == 0:
    with open('mp1/in.dat', 'r') as f1:
        N = int(f1.readline())
        M = int(f1.readline())
else:
    N = 0
    M = 0

# Рассылаем размеры всем процессам
N = comm.bcast(N, root=0)
M = comm.bcast(M, root=0)

# Проверка на корректное число процессов
if numprocs < 2:
    if rank == 0:
        print("Ошибка: программа должна запускаться хотя бы с 2 процессами")
    exit()

# Вычисляем размеры блоков
if rank == 0:
    ave, res = divmod(M, numprocs - 1)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    rcounts[0] = 0
    displs[0] = 0
    for k in range(1, numprocs):
        rcounts[k] = ave + 1 if k < 1 + res else ave
        displs[k] = displs[k - 1] + rcounts[k - 1]
else:
    rcounts = None
    displs = None

# Рассылаем каждому процессу, сколько строк он обрабатывает
M_part = array(0, dtype=int32)
comm.Scatter([rcounts, MPI.INT], [M_part, MPI.INT], root=0)

# Подготавливаем локальный кусок матрицы A
A_part = empty((M_part, N), dtype=float64)

if rank == 0:
    # Читаем всю матрицу A
    A = empty((M, N), dtype=float64)
    with open('mp1/AData.dat', 'r') as f2:
        for j in range(M):
            for i in range(N):
                A[j, i] = float64(f2.readline())
    # Рассылаем блоки A
    comm.Scatterv([A, rcounts * N, displs * N, MPI.DOUBLE],
                  [A_part, M_part * N, MPI.DOUBLE], root=0)
else:
    comm.Scatterv([None, None, None, None],
                  [A_part, M_part * N, MPI.DOUBLE], root=0)

# Вектор x (рассылаем всем)
x = empty(N, dtype=float64)
if rank == 0:
    with open('mp1/xData.dat', 'r') as f3:
        for i in range(N):
            x[i] = float64(f3.readline())

comm.Bcast([x, N, MPI.DOUBLE], root=0)

# Локное произведение
b_part = dot(A_part, x)

# Собираем результат
if rank == 0:
    b = empty(M, dtype=float64)
else:
    b = None

comm.Gatherv([b_part, M_part, MPI.DOUBLE],
             [b, rcounts, displs, MPI.DOUBLE], root=0)

# Вывод результата
if rank == 0:
    with open('Results.dat', 'w') as f4:
        for val in b:
            f4.write(f"{val}\n")
    print(b)
