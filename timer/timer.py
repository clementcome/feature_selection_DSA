import time
import functools
import logging
import os
from datetime import datetime


def timer(func):
    log_file = os.getenv("LOG_FILE", "time.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(message)s",
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Execution of {func.__name__} in {duration:.2f} s")
        return value

    return wrapper


if __name__ == "__main__":

    @timer
    def wait():
        time.sleep(1)
        return "Hello World"

    res = wait()
    print(res)
