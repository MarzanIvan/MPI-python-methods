# debi tirar mas fotos

from mpi4py import MPI
from numpy import arange, empty, array, int32, float64, dot

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    M = 20
    a = arange(1, M + 1, dtype=float64)
else:
    a = None

if rank == 0:
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
else:
    rcounts = None
    displs = None

M_part = array(0, dtype=int32)

comm.Scatter([rcounts, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

a_part = empty(M_part, dtype=float64)

comm.Scatterv([a, rcounts, displs, MPI.DOUBLE],
              [a_part, M_part, MPI.DOUBLE], root=0)

ScalP_temp = empty(1, dtype=float64)

ScalP_temp[0] = dot(a_part, a_part)

ScalP = array(0, dtype=float64)

comm.Reduce([ScalP_temp, 1, MPI.DOUBLE],
            [ScalP, 1, MPI.DOUBLE], op=MPI.SUM, root=0)

# comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE],
#                 [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)

print('ScalP = {0:6.1f} on process {1}'.format(ScalP, rank))

