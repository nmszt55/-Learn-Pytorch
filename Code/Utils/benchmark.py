import time


class BenchMark(object):
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(self.title + ":", "%.3f sec" % (time.time() - self.start_time))