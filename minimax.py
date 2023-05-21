import numpy as np

from state.state import *
import time

_elapsed_time=0

def minimax_init(state:State) -> (np.float16,State):
    global _elapsed_time
    maximize=state.player_to_move==1
    coin_count=np.sum(np.abs(state.matrix))
    depth=6
    if coin_count<=7:
        depth=6
    elif coin_count<=13:
        depth=5
    elif coin_count<=46:
        depth=4
    elif coin_count<=56:
        depth=6
    else:
        depth=8
    #_elapsed_time=time.time()
    return minimax(state,depth,maximize)


def minimax(state:State,depth:int=6,maximize:bool=True,
            alpha:np.float16=np.float16('-inf'),beta:np.float16=np.float16('inf')) -> (np.float16,State):
    global _elapsed_time
    retState:State =None
    if maximize:
        maxEval = np.float16('-inf')
        for child in state_children(state):
            evaluation=alphabeta(child,depth-1,False,alpha,beta)
            maxEval=np.maximum(maxEval,evaluation)
            alpha=np.maximum(alpha,evaluation)
            if beta<=alpha: break
            if maxEval == evaluation: retState = child
        return maxEval,retState
    else:
        minEval=np.float16('inf')
        for child in state_children(state):
            evaluation=alphabeta(child,depth-1,True,alpha,beta)
            minEval=np.minimum(minEval,evaluation)
            beta = np.minimum(beta,evaluation)
            if minEval == evaluation: retState = child
            if beta<=alpha: break
        return minEval,retState


def alphabeta(state:State,depth:int=6, maximize:bool=True,
            alpha:np.float16=np.float16('-inf'),beta:np.float16=np.float16('inf')) -> np.float16:
    if depth==0 or state.is_game_ended():
        return heuristics(state)


    if maximize:
        maxEval = np.float16('-inf')
        for child in state_children(state):
            #if time.time()-_elapsed_time>3: break
            evaluation=alphabeta(child,depth-1,False,alpha,beta)
            maxEval=np.maximum(maxEval,evaluation)
            alpha=np.maximum(alpha,evaluation)
            if beta<=alpha: break
        return maxEval
    else:
        minEval=np.float16('inf')
        for child in state_children(state):
            #if time.time()-_elapsed_time>3: break
            evaluation=alphabeta(child,depth-1,True,alpha,beta)
            minEval=np.minimum(minEval,evaluation)
            beta = np.minimum(beta,evaluation)
            if beta<=alpha: break
        return minEval



if __name__=="__main__":
    pass