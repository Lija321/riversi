from state.state import *
import os
from common import constants
import time
import minimax
import sys
import readline

class Game(object):

    def __init__(self,type:int=1,fen:str=constants.DEFAULT_FEN,player_to_move:int=1):
        """

        :param fen:
        :param player_to_move:
        :param type: 1: AI v AI; 2: P v AI; 3: AI v P; 4: P v P
        """
        self.state=State(None,fen,player_to_move)
        self.type=type
        if not self.type in range(0,5): raise Exception('Invalid type')

    def play(self):
            if self.type==1: self.play_type1()
            elif self.type==2: self.play_type2()
            elif self.type==3: self.play_type3()
            elif self.type==4: self.play_type4()
            evaluation=heuristics(self.state)
            if evaluation>0: print("BLACK WON!!!")
            elif evaluation<0: print("WHITE WON!!!")
            else: print("DRAW.")


    def play_type1(self):
        print_pos_to_console(self.state)
        while not self.state.is_game_ended():
            dt = time.time()
            evaluation, self.state = minimax.minimax(self.state, 6, True, self.state.player_to_move == 1, np.float16('-inf'),
                                                np.float16('inf'))
            print_pos_to_console(self.state)
            print(len(minimax.lookup))
            print(f"{time.time() - dt:.5f}")

    def play_type2(self):
        first_iter=True
        while not self.state.is_game_ended():
            print_pos_to_console(self.state)
            if not first_iter:
                print(f"\nAlgorithm time: {minimax_time:.3f}s")
            else: first_iter=False
            pos=possible_moves(self.state)
            temp=[]
            for i,item in enumerate(pos):
                x,y=item
                temp.append(item)
                print(f"{i+1}. {x+1}{chr(y+65)}")
            choice=str(input(">> "))
            if not(choice.isnumeric() and int(choice)-1 in range(0,len(pos))): continue
            choice=temp[int(choice)-1]
            place_piece(self.state,choice[0],choice[1])
            print_state_to_console(self.state)
            minimax_time = time.time()
            evaluation, self.state = minimax.minimax(self.state, 6, True, self.state.player_to_move == 1, np.float16('-inf'),
                                                np.float16('inf'))
            minimax_time = time.time() - minimax_time

    def play_type3(self):
        while not self.state.is_game_ended():
            minimax_time=time.time()
            evaluation, self.state = minimax.minimax(self.state, 6, True, self.state.player_to_move == 1, np.float16('-inf'),
                                                np.float16('inf'))
            minimax_time=time.time()-minimax_time

            while True:
                print_pos_to_console(self.state)
                print(f"\nAlgorithm time: {minimax_time:.3f}s")
                pos = possible_moves(self.state)
                temp = []
                for i, item in enumerate(pos):
                    x, y = item
                    temp.append(item)
                    print(f"{i + 1}. {x + 1}{chr(y + 65)}")

                choice = str(input(">> "))
                if not (choice.isnumeric() and int(choice) - 1 in range(0, len(pos))): continue
                break
            choice = temp[int(choice) - 1]
            place_piece(self.state, choice[0], choice[1])
            print_state_to_console(self.state)

    def play_type4(self):
        while not self.state.is_game_ended():
            print_pos_to_console(self.state)
            pos = possible_moves(self.state)
            temp = []
            for i, item in enumerate(pos):
                x, y = item
                temp.append(item)
                print(f"{i + 1}. {x + 1}{chr(y + 65)}")
            choice = str(input(">> "))
            if not (choice.isnumeric() and int(choice) - 1 in range(0, len(pos))): continue
            choice = temp[int(choice) - 1]
            place_piece(self.state, choice[0], choice[1])
            print_state_to_console(self.state)

def print_state_to_console(state,evaluation=None):
    os.system('clear')
    header = "   | 🇦  | 🇧  | 🇨  | 🇩  | 🇪  | 🇫  | 🇬  | 🇭  |"
    print(header)
    print("-" * (len(header)))
    for i in range(8):
        print(" " + f"{i + 1} ", end="")
        for j in range(8):
            c=state.matrix[i,j]
            if c == 1:
                c = "⚫️"
            elif c == -1:
                c = "⚪️"
            elif c == 2:
                if state.player_to_move == 1:
                    c = "◾️"
                else:
                    c = "◽️"
            else:
                c = " "
            print(f"|{c:^4}", end='')
        print("|")
        print('-' * (len(header)))
    if not evaluation is None:
        print(evaluation)

def print_pos_to_console(state):
    state_copy=State(state)
    pos=possible_moves(state_copy)
    for x,y in pos:
        state_copy.matrix[x,y]=2
    print_state_to_console(state_copy)

