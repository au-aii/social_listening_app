# app/utils.py (または各モジュール内)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# logger.info("This is an info message")
# logger.error("An error occurred")