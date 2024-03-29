import os
import sys
import traceback
import pickle

from game.game import Game
import state

def main():
    #global lookup
    #with open('lookup.pkl', 'rb') as f:
    #    state.state.lookup = pickle.load(f)

    while True:
        os.system("clear")
        print("Choose mode:\n1. AI vs AI\n2. P vs AI\n3. AI v P\n4. P v P")
        mode=str(input(">> "))
        if not(mode.isnumeric() and int(mode) in range(0,5)):continue
        mode=int(mode)
        break
    g=Game(mode,None,None,None)
    g.play()
    #with open('lookup.pkl', 'wb') as f:
    #    pickle.dump(state.state.lookup, f)


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

    except Exception:
        print(traceback.format_exc())
    finally:
        sys.exit(-1)