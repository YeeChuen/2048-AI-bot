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
    print("............................. MCTS play()")
    #up:0, right:1, down:2, left:3
    movetranslate={0:"up", 1:"right", 2:"down", 3:"left"}

    state = State2048(boardSize=4)
    monte_carlo = MCTS(state, no_simulation=500, depth = 6)
    #monte_carlo.print()

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
    print(".............................")
    return [int(state.score), int(count)]

# ______________________________________________________________________________
# Random play
# a random bot that picks action on random
def randomplay():
    print("............................. random play()")

    state = State2048(boardSize=4)
    #TOBE DELETED
    onetime = False
    count=1
    # state.checkGameOver()
    while state.checkGameOver() is False:
        #print("====== new board (round {})=====".format(str(count)))
        #monte_carlo.currnode.state2048.print()
        #print("current board score: {}".format(str(state.score)))
        action_list = state.actions()
        action = np.random.choice(action_list)
        state = state.move(action)
        #TOBE DELETED
        onetime = True
        count+=1

    state.print()
    print("Game Over, AI score: {}".format(str(state.score)))
    print("Game Over, total round: {}".format(str(count)))
    print(".............................")
    return [int(state.score), int(count)]

# ______________________________________________________________________________
# Greedy play
# Greedy bot that always chooses the greedy action to maximize score
def greedyplay():
    print("............................. greedy play()")

    state = State2048(boardSize=4)
    monte_carlo = MCTS(state, no_simulation=500, depth = 6)

    #TOBE DELETED
    onetime = False
    count=1
    # state.checkGameOver()
    while state.checkGameOver() is False:
        #print("====== new board (round {})=====".format(str(count)))
        #monte_carlo.currnode.state2048.print()
        #print("current board score: {}".format(str(state.score)))
        action = monte_carlo.greedy_action(monte_carlo.currnode, monte_carlo.currnode.state2048.actions())
        monte_carlo.update_currnode(action)
        state = monte_carlo.currnode.state2048
        #TOBE DELETED
        onetime = True
        count+=1

    state.print()
    print("Game Over, AI score: {}".format(str(state.score)))
    print("Game Over, total round: {}".format(str(count)))
    print(".............................")
    return [int(state.score), int(count)]
# ______________________________________________________________________________
# Report function
# function that reports the result based on a sample size
def report_result(sample_size, func, bot_type):  
    print("----------------- {} sample simulation starts -----------------".format(bot_type))
    report_countlist = []
    report_scorelist = []
    for _ in range(sample_size):
        list = func()
        report_scorelist.append(list[0])
        report_countlist.append(list[1])
    
    print("_________________________________________")
    print("{} result on average out of {} games".format(bot_type, str(sample_size)))
    print("average score: {}".format(int(sum(report_scorelist)/len(report_scorelist))))
    print("average round: {}".format(int(sum(report_countlist)/len(report_countlist))))
    print("------------------ {} sample simulation end ------------------".format(bot_type))

# ______________________________________________________________________________
# Main
if __name__ == "__main__":
    #Random bot
    report_result(10, randomplay, "Random Bot")

    #Greedy bot
    report_result(10, greedyplay, "Greedy Bot")

    #MCTS bot
    report_result(10, MCTSplay, "MCTS Bot")

