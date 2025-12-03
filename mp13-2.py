from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numprocs = comm.Get_Size()

# Разделение данных между процессами
A_local = cp.load(f"AData_part_{rank}.dat")
b_local = cp.load(f"bData_part_{rank}.dat")
x_local = cp.zeros_like(b_local)

for iter in range(max_iter):
    r_local = b_local - cp.dot(A_local, x_local)
    p_local = r_local.copy()
    Ap_local = cp.dot(A_local, p_local)

    # Глобальные редукции через MPI
    alpha_num = cp.dot(r_local, r_local).get()
    alpha_den = cp.dot(p_local, Ap_local).get()
    alpha_num = comm.allreduce(alpha_num, op=MPI.SUM)
    alpha_den = comm.allreduce(alpha_den, op=MPI.SUM)
    alpha = alpha_num / alpha_den

    x_local += alpha * p_local
    r_new = r_local - alpha * Ap_local
    if cp.linalg.norm(r_new) < eps:
        break

comm.Barrier()
if rank == 0:
    print("Calculation finished")
