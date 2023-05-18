from state.state import State
from common import constants
import numpy as np

class HashMap(object):

    def __init__(self,moduo=10000000000):
        self.moduo=moduo

    def state_hash_code(self,state:State) -> int: #TODO moguce u c
        hash_value = hash(tuple(map(tuple, state.matrix.tolist())))
        return hash_value%self.moduo

def _main():
    pass
if __name__=="__main__":
    _main()