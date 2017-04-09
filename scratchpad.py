

'''
#newgame =
BLANK = 0
width = 7
height = 7

_board_state = [BLANK] * (width * height + 3)
print(_board_state)
print(len(_board_state))

print( 3  + 3 * 7)

print([1,2,3,4,5][-2])

bn = 1>2 or 2>3 or 3<4
print (bn)

print(0^0)
print(0^1)
print(1^0)
print(1^1)



print ([(i, j) for j in range(7) for i in range(7)])



from isolation import Board
from sample_players import *

# create an isolation board (by default 7x7)
player1 = HumanPlayer()
player2 = GreedyPlayer()
game = Board(player1, player2)

winner, history, outcome = game.play()

print(winner)
print( outcome)

print(game._board_state)

h=11
w=11
center = (int(h/2), int(w/2))
valid = [(1,1), (2,2), (3,3), (4,4), (5,5)]
if  center in valid:
    print("yes")
    print (center)

'''