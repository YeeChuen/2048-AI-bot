# Author: Yee Chuen Teoh
# Project: 572 Term Project
# Title: main.py
# Description: main file to run 2048
# Reference: none

from search import *
from Game2048 import *
from MonteCarlo import *

# ______________________________________________________________________________
# User play

#For user to play 2048/test implementation
# taken from Matt's Game2048 file. 
def userplay():
    state = State2048(boardSize=4)
    while state.checkGameOver() is False:
        #os.system('cls||clear')
        print("")
        state.print()
        print("current score: {}".format(str(state.score)))
        print("")
        print("enter 'w', 'd', 's', or 'a' to move up, right, down, or left respectively")
        m = str(input())
        d = None
        if m == "w":
            d = 0
        if m == "d":
            d = 1
        if m == "s":
            d = 2
        if m == "a":
            d = 3

        if d is not None:
            newState = state.move(d)

            if newState is not None:
                state = newState

    state.print()
    print("Game Over, you scored")
    print(state.score)

# ______________________________________________________________________________
# MCTS play
# implementation of MCTS to 2048

def MCTSplay():
    #up:0, right:1, down:2, left:3
    movetranslate={0:"up", 1:"right", 2:"down", 3:"left"}

    state = State2048(boardSize=4)
    while state.checkGameOver() is False:
        print("")
        state.print()
        print("current score: {}".format(str(state.score)))
        print("")

        action=monte_carlo_tree_search(state)

        if action is not None:
            print("MCTS chosen action {}".format(movetranslate[action]))
            newState = state.move(action)

            if newState is not None:
                state = newState

    state.print()
    print("Game Over, AI scored")
    print(state.score)


if __name__ == "__main__":
    #randomPlay()
    userplay()
    #MCTSplay()

#____________________________________________________________________________________
#TOBE DELETED Section
#Test Code

#Test node class (add child, traversal, backprop, etc)
    '''
    #test code
    newstate = State2048(boardSize=4)
    #print(newstate.__hash__())
    #newstate.print()
    rootnode = node(state2048=newstate)
    
    # creating first child for root
    childnode1 = rootnode.addchild(0)

    # creating second child for root
    childnode2 = rootnode.addchild(1)

    # creating child for child1
    leafnode1 = childnode1.addchild(1)

    # creating child for child2
    leafnode2 = childnode2.addchild(1)

    #add score for leafnode
    leafnode1.score = 100
    leafnode2.score = 999

    # creating third child for root
    childnode3 = rootnode.addchild(0)

    currnode = leafnode1
    while currnode.parent:
        print("run")
        currnode = currnode.backprop()    

    currnode = leafnode2
    while currnode.parent:
        print("run")
        currnode = currnode.backprop()

    print("")
    print("CURRENT NODE INFORMATION, should be the root with score")
    currnode.print()
    rootnode.print()
    childnode1.print()
    childnode2.print()
    childnode3.print()
    leafnode1.print()
    leafnode2.print()

    '''

#Test MCTS class
gamestate = State2048(boardSize=4)
monte_carlo = MCTS(gamestate, no_simulation=4, depth = 4)
monte_carlo.print()
print("====== SIMULATION =====")
monte_carlo.simulation()

