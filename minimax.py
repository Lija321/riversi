import numpy as np

from state.state import *
import time

_elapsed_time=0
_vremesnko_ogranicenje = 3  # sekunnde


def minimax_init(state:State) -> (np.float16,State):
    global _elapsed_time
    global _vremesnko_ogranicenje
    maximize=state.player_to_move==1
    _elapsed_time=time.time()
    coins_left=(64-np.sum(np.abs(state.matrix)))
    depth=min(coins_left,3)
    best_state:State=None
    best_eval=0
    while time.time()-_elapsed_time<_vremesnko_ogranicenje:
        evaluation,new_state=minimax(state, depth, maximize)
        if not new_state is None and depth<=coins_left:
            best_state=new_state
            best_eval=evaluation
            depth+=1
        else: break
    if best_state is None:
        _elapsed_time=time.time()
        best_eval,best_state=minimax(state,3,maximize)
    return best_eval,best_state,depth
    #coin_count=np.sum(np.abs(state.matrix))
    #depth=6
    #if coin_count<=7:
    #    depth=6
    #elif coin_count<=20:
    #    depth=5
    #elif coin_count<=46:
    #    depth=4
    #elif coin_count<=56:
    #    depth=6
    #else:
    #    depth=8
    ##_elapsed_time=time.time()
    #return minimax(state,depth,maximize)


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
            if beta<=alpha:
                #print('pruned max')
                break
            if maxEval == evaluation: retState = child
        if time.time() - _elapsed_time > _vremesnko_ogranicenje:
            return 0.0, None
        return maxEval,retState
    else:
        minEval=np.float16('inf')
        for child in state_children(state):
            if time.time()-_elapsed_time>_vremesnko_ogranicenje:
                return 0.0,None
            evaluation=alphabeta(child,depth-1,True,alpha,beta)
            minEval=np.minimum(minEval,evaluation)
            beta = np.minimum(beta,evaluation)
            if minEval == evaluation: retState = child
            if beta<=alpha:
                #print('pruned min')
                break
        if time.time() - _elapsed_time > _vremesnko_ogranicenje:
            return 0.0, None
        return minEval,retState


def alphabeta(state:State,depth:int=6, maximize:bool=True,
            alpha:np.float16=np.float16('-inf'),beta:np.float16=np.float16('inf')) -> np.float16:
    global _elapsed_time

    if depth==0 or state.is_game_ended():
        return heuristics(state)


    if maximize:
        maxEval = np.float16('-inf')
        for child in state_children(state):
            #if time.time()-_elapsed_time>3: break
            if time.time()-_elapsed_time>_vremesnko_ogranicenje:
                return 0.0

            evaluation=alphabeta(child,depth-1,False,alpha,beta)
            maxEval=np.maximum(maxEval,evaluation)
            alpha=np.maximum(alpha,evaluation)
            if beta<=alpha:
                #print('pruned max')
                break
        return maxEval
    else:
        minEval=np.float16('inf')
        for child in state_children(state):
            #if time.time()-_elapsed_time>3: break
            if time.time()-_elapsed_time>_vremesnko_ogranicenje:
                return 0.0
            evaluation=alphabeta(child,depth-1,True,alpha,beta)
            minEval=np.minimum(minEval,evaluation)
            beta = np.minimum(beta,evaluation)
            if beta<=alpha:
                #print('pruned min')
                break
        return minEval

if __name__=="__main__":
    pass