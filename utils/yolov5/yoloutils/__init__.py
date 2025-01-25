import contextlib
import platform
import threading


def emojis(str=""):
    """..."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str


class TryExcept(contextlib.ContextDecorator):
    def __init__(
        self,
        msg="",
    ):
        """..."""
        self.msg = msg

    def __enter__(self):
        """..."""
        pass

    def __exit__(
        self,
        exc_type,
        value,
        traceback,
    ):
        """..."""
        if value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    """..."""

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def join_threads(verbose=False):
    """..."""
    main_thread = threading.current_thread()
    for t in threading.enumerate():
        if t is not main_thread:
            if verbose:
                print(f"Joining thread {t.name}")
            t.join()
