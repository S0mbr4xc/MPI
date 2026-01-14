from mpi4py import MPI
import sys
from master_node import MasterNode
from child_node import ChildNode

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            print("Error: This program requires at least 2 processes (1 Master, 1 Child).")
            print("Run with: mpiexec -n 2 python main.py")
        sys.exit(1)

    if rank == 0:
        # Master Node
        app = MasterNode(comm)
        app.run()
    else:
        # Child Node
        node = ChildNode(rank)
        # Wait for data from master
        try:
            data = comm.recv(source=0, tag=11)
            result = node.train(data)
            comm.send(result, dest=0, tag=22)
        except Exception as e:
            print(f"[Rank {rank}] Error: {e}")

if __name__ == "__main__":
    main()
