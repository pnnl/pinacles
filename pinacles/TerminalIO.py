from mpi4py import MPI


def start_message():
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Hello from PINACLES!")
        print("\t Today I am running on ", MPI.COMM_WORLD.Get_size(), " MPI ranks!")
    return


def end_message():
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("PINACLES is all done!")
        print("\t Good Day!")
    return


def root_print(output, tab_level=0):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("\t" * tab_level, output)

    return
