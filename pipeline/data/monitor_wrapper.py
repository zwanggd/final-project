import time
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def monitor_data_flow(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        log.info(f"Entering {func.__name__}")
        
        # Log input data characteristics
        for i, arg in enumerate(args):
            log.info(f"Input {i}: Type={type(arg)}, Shape={getattr(arg, 'shape', None)}")
        for key, val in kwargs.items():
            log.info(f"Input {key}: Type={type(val)}, Shape={getattr(val, 'shape', None)}")
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Log output data characteristics
        log.info(f"Output: Type={type(result)}, Shape={getattr(result, 'shape', None)}")
        log.info(f"Execution time for {func.__name__}: {time.time() - start_time:.4f} seconds")
        
        return result
    return wrapper
