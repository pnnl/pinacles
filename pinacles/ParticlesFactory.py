import pinacles.ParticlesGrid as ParticlesGrid
from pinacles import UtilitiesParallel


class ParticlesDummy:
    def __init__(self):

        self.frequency = 1e9

        return

    def update(self):

        return

    def output(self):

        return

    def restart(self):
        
        return
    
    def dump_restart(self, data_dict):
        
        return

def ParticlesFactory(
    namelist,
    Grid,
    Ref,
    TimeSteppingController,
    VelocityState,
    ScalarState,
    DiagnosticState,
):
    try:
        import h5py
    except:
        UtilitiesParallel.print_root("\t No H5PY--Disabling Particles.")
        return ParticlesDummy()


    if "particles" in namelist:

        assert "type" in namelist["particles"]

        if namelist["particles"]["type"].upper() == "SIMPLE":
            UtilitiesParallel.print_root("\tUsing Particles Simple.")
            return ParticlesGrid.ParticlesSimple(
                namelist,
                Grid,
                Ref,
                TimeSteppingController,
                VelocityState,
                ScalarState,
                DiagnosticState,
            )

    UtilitiesParallel.print_root(
        "\t This simulation will not be using particles returning Particles Dummy."
    )

    return ParticlesDummy()
