
class Plume:

    def __init__(self, location, start_time, Grid, ScalarState, TimeSteppingController):

        
        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController
        self._ScalarState = ScalarState

        self._location = location
        self._start_time = start_time

        return

    def update(self):

        return
    
    @property
    def location(self):
        return self._location

    @property
    def start_time(self):
        return self._start_time

class Plumes:

    def __init__(self, namelist, Grid, ScalarState, TimeSteppingController):

        self._Grid = Grid
        self._TimeSteppingController = TimeSteppingController
        self._ScalarState = ScalarState
        self._locations = None
        self._startimes = None

        self._n = 0

        if 'plumes' in namelist:
            self._locations  = namelist['plumes']['locations']
            self._startimes = namelist['plumes']['starttimes']
        else: 
            return

        assert len(self._locations) == len(self._startimes)


        self._list_of_plumes = []
        for loc, start in zip(self._locations, self._startimes):
            self._list_of_plumes.append(Plume(loc, start, self._Grid, self._ScalarState, self._TimeSteppingController))


        return

    def update(self):

        if self._n == 0:
            # If there ae no plumes, just return
            return


        return