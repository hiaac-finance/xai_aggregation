import pathos


# redefine process pool via inheritance
import multiprocess.context as context
class NoDaemonProcess(context.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

import multiprocessing
class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# class NoDaemonProcessPool(pathos.multiprocessing.Pool):
#     def Process(self, *args, **kwds):
#         return NoDaemonProcess(*args, **kwds)

class NoDaemonProcessPool(pathos.multiprocessing.ProcessingPool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NoDaemonProcessPool, self).__init__(*args, **kwargs)