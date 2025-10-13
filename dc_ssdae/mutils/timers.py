

import time

####################################################################
# Timers
####################################################################


class Timer:
    def __init__(self, ms=False, elapsed=0):
        self.elapsed = elapsed
        self.start_at = None
        self.show_ms = ms  # Show milli-seconds

    def start(self, reset=False):
        """Start the timer. It should always be called when the timer is not currently running, except if used with reset=True

        Args:
            reset (bool, optional): Reset the total elapsed time to 0.
                If the timer was running, is will start again from the curent time.
                Defaults to False.
        """
        if reset:
            self.reset()
        else:
            assert self.start_at is None, "Timer is already in use"
        self.start_at = time.time()

    def reset(self):
        self.elapsed = 0
        if self.start_at:
            self.start_at = time.time()

    def stop(self):
        self.elapsed += time.time() - self.start_at
        self.start_at = None
        return self.get()

    def running(self):
        return self.start_at is not None

    def get(self):
        if self.running():
            return self.elapsed + time.time() - self.start_at
        return self.elapsed

    @staticmethod
    def format_time(t, ms=False):
        hours, t = divmod(t, 3600)
        minutes, seconds = divmod(t, 60)
        millis_str = ""
        if ms:
            millis = (seconds - int(seconds)) * 1000
            millis_str = f".{int(millis):03}"
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}{millis_str}"

    def s(self, digits=3):
        fmt = "{0:." + str(digits) + "f}s"
        return fmt.format(self.get())

    def ms(self):
        return self.format_time(self.get(), True)

    def __str__(self) -> str:
        return self.format_time(self.get(), self.show_ms)

    def __repr__(self) -> str:
        return self.__str__()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __call__(self, reset=False):
        if reset:
            self.reset()
        return self


class TimersManager:
    def __init__(self, *timer_list):
        self.add(*timer_list)

    def add(self, *names):
        for name in names:
            if not hasattr(self, name):
                self.__setattr__(name, Timer())

    def get_timer_list(self):
        return [attr for attr in self.__dir__() if isinstance(self[attr], Timer)]

    def __getitem__(self, attr):
        return self.__getattribute__(attr)

    def join_str(self, sep=" "):
        return sep.join([f"T_{name}={self[name]}" for name in self.get_timer_list()])

    def __str__(self) -> str:
        return self.join_str()

    def __repr__(self) -> str:
        return self.__str__()

    def state_dict(self):
        return {name: self[name].get() for name in self.get_timer_list()}

    def sum(self):
        total_s = sum([self[name].get() for name in self.get_timer_list()])
        return Timer(elapsed=total_s)

    def load_state_dict(self, state):
        for name, val in state.items():
            self[name].elapsed = val
