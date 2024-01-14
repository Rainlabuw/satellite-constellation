class Task(object):
    def __init__(self, loc, benefit):
        """
        loc is a poliastro.SpheroidLocation which encodes the position on Earth of the task.
        benefit is a T size array of benefits of the task over time.
        """
        self.loc = loc
        self.benefit = benefit

        self.id = None