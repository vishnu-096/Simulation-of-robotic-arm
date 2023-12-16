import logging

logging_level = logging.INFO
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the default logging level to INFO
setup_logging(logging_level)
