from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_Size()

# Загрузка данных
A_gpu = cp.load( &apos;AData.dat&apos;)
b_gpu = cp.load( &apos;bData.dat & apos;)
x_gpu = cp.zeros_like(b_gpu)

# Инициализация векторов
r_gpu = b_gpu - cp.dot(A_gpu, x_gpu)
p_gpu = r_gpu.copy()

for k in range(max_iter):
    Ap_gpu = cp.dot(A_gpu, p_gpu)
    alpha = cp.dot(r_gpu, r_gpu) / cp.dot(p_gpu, Ap_gpu)
    x_gpu += alpha * p_gpu
    r_new = r_gpu - alpha * Ap_gpu
    beta = cp.dot(r_new, r_new) / cp.dot(r_gpu, r_gpu)
    p_gpu = r_new + beta * p_gpu
    r_gpu = r_new
    if cp.linalg.norm(r_gpu) < eps:
        break
