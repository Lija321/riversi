import numpy as np

from state.state import *


global lookup
lookup={}

def minimax(state:State,depth:int=6,first_call:bool=True, maximize:bool=True,
            alpha:np.float16=np.float16('-inf'),beta:np.float16=np.float16('inf')) -> (np.float16,State):
    if depth==0 or state.is_game_ended():
        if not state in lookup:
            lookup[state]=heuristics(state)
        return lookup[state], None

    retState:State =None

    if maximize:
        maxEval = np.float16('-inf')
        for child in state_children(state):
            evaluation,_=minimax(child,depth-1,False,False,alpha,beta)
            maxEval=np.maximum(maxEval,evaluation)
            alpha=np.maximum(alpha,evaluation)
            if beta<=alpha: break
            if maxEval==evaluation: retState=child #TODO Ekspreimentisi sa razlicitm biranjem ovoga, NPR razlicita f-ja za prvi sloj
        if not first_call:return maxEval,None
        else:return maxEval,retState
    else:
        minEval=np.float16('inf')
        for child in state_children(state):
            evaluation,_=minimax(child,depth-1,False,True,alpha,beta)
            minEval=np.minimum(minEval,evaluation)
            beta = np.minimum(beta,evaluation)
            if beta<=alpha: break
            if minEval==evaluation: retState=child
        if not first_call:return minEval,None
        else:return minEval,retState


if __name__=="__main__":
    pass