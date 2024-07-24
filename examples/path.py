import sys
import os

def add_src_to_path():
    # Add the src directory to the Python path
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
    if src_path not in sys.path:
        sys.path.append(src_path)
:wq
