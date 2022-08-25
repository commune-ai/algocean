import datetime

def isoformat2datetime(isoformat:str):
    dt, _, us = isoformat.partition(".")
    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    us = int(us.rstrip("Z"), 10)
    dt = dt + datetime.timedelta(microseconds=us)
    assert isinstance(dt, datetime.datetime)
    return dt

def isoformat2timestamp(isoformat:str, return_type='int'):
    supported_types = ['int', 'float']
    assert return_type in supported_types, f'return type should in {supported_types} but you put {return_type}'
    dt = isoformat2datetime(isoformat)
    timestamp = eval(return_type)(dt.timestamp())
    assert isinstance(timestamp, int)
    return timestamp


def timedeltatimestamp( **kwargs):
    assert len(kwargs) == 1
    supported_modes = ['hours', 'seconds', 'minutes', 'days']
    mode = list(kwargs.keys())[0]
    assert mode in supported_modes, f'return type should in {supported_modes} but you put {mode}'
    
    current_timestamp = datetime.datetime.utcnow()
    timetamp_delta  =  current_timestamp.timestamp() -  ( current_timestamp- datetime.timedelta(**kwargs)).timestamp()
    return timetamp_delta


class Timer:
    supported_modes = ['second', 'timestamp']


    def __init__(self, start=False):
        self.start_time = None
        self.end_time = None
        if start:
            self.start()
    
    def start(self):
        assert self.start_time == None, f'You already started the timer at {self.start_time}'
        
        self.start_time = self.current_timestamp


    @staticmethod
    def get_current_timestamp():
        return int(datetime.datetime.utcnow().timestamp())

    @property
    def current_timestamp(self):
        return self.get_current_timestamp()

    def elapsed_time(self, return_type='second'):
        
        assert isinstance(self.start_time, int), f'You need to start the timer with self.start()'
        assert return_type in self.supported_modes, f'return_type: {return_type} not supported in {self.supported_modes}'
        
        timestamp_period =  self.current_timestamp -self.start_time 
        return timestamp_period

    def stop(self):
        self.end_time = None
        self.start_time = None
