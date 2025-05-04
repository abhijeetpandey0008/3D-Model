utils.py
import os
import logging

logging.basicConfig(level=logging.INFO)

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
