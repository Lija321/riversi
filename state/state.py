import numpy as np

from common.exceptions import *
from common import constants
import c_extentions.riversi_c_utils as c_util


class State(object):

    def __init__(self, state: __build_class__ = None, fen: str = None, player_to_move: int = 1):
        if not state is None:
            self.matrix: np.matrix = np.matrix(state.matrix)
            self.player_to_move: np.int8 = state.player_to_move
        elif not fen is None:
            self.matrix: np.matrix = fen_to_matrix(fen)
            self.player_to_move: np.int8 = np.int8(player_to_move)
        else:
            raise StateException()

    def is_game_ended(self):
        return len(possible_moves(self)) == 0

    def __str__(self):
        return str(self.matrix)

    def __hash__(self):
        return hash(str(self.matrix))


const = {
    'b': np.int8(1),
    'w': np.int8(-1)
}

directions = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)

lookup = {}

posible_moves_lookup = {}


def fen_to_matrix(fen: str) -> np.matrix:
    data: np.matrix = np.matrix(np.zeros((8, 8), dtype=np.int8))
    fen.lower()
    fen = fen.split('/')
    for i in range(8):
        if fen[i] == '8':
            continue
        else:
            counter = 0
            j = -1
            while counter < 8:
                j += 1
                x = fen[i][j]
                if x.isnumeric():
                    counter += int(x)
                elif x == 'b' or x == 'w':
                    data[i, counter] = const[x]
                    counter += 1
                else:
                    raise FenException("Greska u fenu")
    return data


def possible_moves(state: State):
    if state in posible_moves_lookup: return posible_moves_lookup[str(state)]
    ret = set()  # TODO Nadji nesto efikasnije
    player_to_move: np.int8 = state.player_to_move  # TODO popravi ovo
    for x in range(8):
        for y in range(8):
            if state.matrix[x, y] != player_to_move: continue
            for dx, dy in directions:
                x1: int = x
                y1: int = y
                line_found: bool = False
                while x1 in range(0, 8) and y1 in range(0, 8):
                    x1 += dx
                    y1 += dy
                    if not (x1 in range(0, 8) and y1 in range(0, 8)): break
                    if line_found:
                        if state.matrix[x1, y1] == -player_to_move:
                            continue
                        elif state.matrix[x1, y1] == player_to_move:
                            break
                        else:
                            ret.add((x1, y1))
                            break
                    else:
                        if state.matrix[x1, y1] != -player_to_move:
                            break
                        else:
                            line_found = True
    posible_moves_lookup[str(state)] = ret
    return ret


"""
def heuristics(state:State) ->np.float16:
    if state in lookup: return lookup[str(state)]

    if state.is_game_ended():
        piece_sum=np.sum(state.matrix)
        if piece_sum==0: return np.float16(0.0)
        if piece_sum>0: return np.float16('inf')
        else: return np.float16('-inf')

    coeficients = np.matrix([
        [50, -1, 4, 4, 4, 4, -1, 50],
        [-1, -1, 2, 2, 2, 2, -1, -1],
        [4, 2, 1, 1, 1, 1, 2, 4],
        [4, 2, 1, 0.2, 0.2, 1, 2, 4],
        [4, 2, 1, 0.2, 0.2, 1, 2, 4],
        [4, 2, 1, 1, 1, 1, 2, 4],
        [-1, -1, 2, 2, 2, 2, -1, -1],
        [50, -1, 4, 4, 4, 4, -1, 50],
    ],np.float16)

    score = np.float16(0.0)
    for i in range(8):
        for j in range(8):
            score+=coeficients[i,j]*state.matrix[i,j]

    lookup[str(state)]=score
    return score
"""


def heuristics(state: State) -> np.float16:
    if state in lookup: return lookup[str(state)]
    if state.is_game_ended():
        piece_sum = np.sum(state.matrix)
        if piece_sum == 0: return np.float16(0.0)
        if piece_sum > 0:
            return np.float16('inf')
        else:
            return np.float16('-inf')
    score = c_util.heuristic(state.matrix)
    lookup[str(state)] = score
    return score


def place_piece(state: State, x, y) -> None:
    state.matrix[x, y] = state.player_to_move
    for_turning = set()
    for dx, dy in directions:
        x1: int = x
        y1: int = y
        line_found: bool = False
        temp = set()
        while x1 in range(0, 8) and y1 in range(0, 8):
            x1 += dx
            y1 += dy
            if not (x1 in range(0, 8) and y1 in range(0, 8)): break
            if line_found:
                if state.matrix[x1, y1] == -state.player_to_move:
                    temp.add((x1, y1))
                    continue
                elif state.matrix[x1, y1] == state.player_to_move:
                    for_turning = for_turning.union(temp)
                    break
                else:
                    break
            else:
                if state.matrix[x1, y1] != -state.player_to_move:
                    break
                else:
                    line_found = True
                    temp.add((x1, y1))

    for x, y in for_turning:
        state.matrix[x, y] = state.player_to_move
    state.player_to_move = -state.player_to_move


def state_children(state: State) -> list:
    pos = possible_moves(state)
    ret = []
    for x, y in pos:
        new_state: State = State(state, None, None)
        place_piece(new_state, x, y)
        ret.append(new_state)
    return ret


def _main():
    state = State(None, constants.DEFAULT_FEN)
    print(str(state))
    print(possible_moves(state))
    x, y = eval(input(" >> "))
    place_piece(state, x, y)
    print(str(state))


if __name__ == "__main__":
    _main()
