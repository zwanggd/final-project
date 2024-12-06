import time
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def timer(module_name=None):
    """
    A decorator to time the execution of a function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            log.info(f"Starting {module_name or func.__name__}")
            
            result = func(*args, **kwargs)
            
            elapsed_time = time.time() - start_time
            log.info(f"Finished {module_name or func.__name__} in {elapsed_time:.4f} seconds")
            return result
        return wrapper
    return decorator
