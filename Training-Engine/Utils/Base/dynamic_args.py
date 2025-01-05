# Libs >>>
import time

# Main >>>
class DynamicArg:
    def __init__(self, callable):
        # Setup self
        self.env_args = {}
        self.callable = callable
        
    def update(self, env_args):
        pass