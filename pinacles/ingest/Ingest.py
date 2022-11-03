from sqlite3 import Time
from unicodedata import name


def IngestFactory(namelist, Grid, TimeSteppingController):
    
    if 'lbc' not in namelist:
        from pinacles.ingest.IngestBase import IngestBase
        return IngestBase(namelist, Grid, TimeSteppingController)
    if 'open_boundary_treatment' not in namelist['lbc']:
        from pinacles.ingest.IngestBase import IngestBase
        return IngestBase(namelist, Grid, TimeSteppingController)
    
    #if namelist['lbc']['open_boundary_treatment'] == "reanalysis":
    #    from pinacles.ingest.IngestERA5 import IngestERA5
    #    return IngestERA5(namelist, Grid, TimeSteppingController)
    
    if namelist['meta']['casename'].upper() == "REAL":
        from pinacles.ingest.IngestWRF import IngestWRF
        return IngestWRF(namelist, Grid, TimeSteppingController)
    else:
        from pinacles.ingest.IngestBase import IngestBase
        return IngestBase(namelist, Grid, TimeSteppingController)
