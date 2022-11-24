# Author: Yee Chuen Teoh
# Project: 572 Term Project
# Title: main.py
# Description: main file to run 2048
# Reference: none

from Game2048 import *
from MonteCarlo import *

# ______________________________________________________________________________
# User play

#For user to play 2048/test implementation
# taken from Matt's Game2048 file. 
def userplay():
    count = 0
    state = State2048(boardSize=4)
    while state.checkGameOver() is False:
        #os.system('cls||clear')
        print("")
        print("====== new board (round {})=====".format(str(count)))
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
        count+=1

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
    monte_carlo = MCTS(state, no_simulation=500, depth = 6)
    monte_carlo.print()

    #TOBE DELETED
    onetime = False
    count=1
    # state.checkGameOver()
    while state.checkGameOver() is False:
        #print("====== new board (round {})=====".format(str(count)))
        #monte_carlo.currnode.state2048.print()
        #print("current board score: {}".format(str(state.score)))
        action = monte_carlo.simulation()
        monte_carlo.update_currnode(action)
        state = monte_carlo.currnode.state2048
        #TOBE DELETED
        onetime = True
        count+=1

    state.print()
    print("Game Over, AI score: {}".format(str(state.score)))
    print("Game Over, total round: {}".format(str(count)))
    return [int(state.score), int(count)]

if __name__ == "__main__":
    #randomPlay()
    #userplay()
    report_countlist = []
    report_scorelist = []
    for _ in range(10):
        list = MCTSplay()
        report_scorelist.append(list[0])
        report_countlist.append(list[1])
    
    print("MCTS result on average out of 10 games")
    print("average score: {}".format(int(sum(report_scorelist)/len(report_scorelist))))
    print("average round: {}".format(int(sum(report_countlist)/len(report_countlist))))


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

