def ScalarAdvectionFactory(namelist): 
    
    if namelist['ScalarAdvection']['name'] == 'ScalarWENO5': 
        return ScalarWENO5

    return 


class ScalarAdvectionBase: 

    def __init__(self): 

        return 

    def update(self): 

        return 


class ScalarWENO5(ScalarAdvectionBase): 
    def __init__(self): 
        ScalarAdvectionBase.__init__(self) 
        return 

    def update(self):

        return 