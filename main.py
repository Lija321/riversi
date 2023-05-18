import os
import sys

from game.game import Game

def main():
    while True:
        os.system("clear")
        print("Choose mode:\n1. AI vs AI\n2. P vs AI\n3. AI v P\n4. P v P")
        mode=str(input(">> "))
        if not(mode.isnumeric() and int(mode) in range(0,5)):continue
        mode=int(mode)
        break
    g=Game(mode)
    g.play()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(e)
    finally:
        sys.exit(-1)