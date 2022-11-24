# Author: Yee Chuen Teoh
# Project: 572 Term Project
# Title: 
# Description: 
# Reference: aimacode from github
# Question: how to best represent MCTS on 2048? the game has no win or lose, single-agent, stochastic

import copy
import itertools
import random
from collections import namedtuple

import numpy as np

# ______________________________________________________________________________
# Game State

GameState = namedtuple('GameState', 'to_move, utility, board, moves')
StochasticGameState = namedtuple('StochasticGameState', 'to_move, utility, board, moves, chance')

# ______________________________________________________________________________
# MCT node

class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""
    # U is the usually the total win
    # N is usually the total simulation 
    def __init__(self, parent=None, state=None, U=0, N=0):
        self.parent=parent
        self.state=state
        self.U=U
        self.N=N
        self.children = {}
        self.actions = None


def ucb(n, C=1.4):
    return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)


# ______________________________________________________________________________
# Monte Carlo Tree Search


def monte_carlo_tree_search(gamestate, N=100):

    #TOBE DELETED ----------
    #print("--- MCTS starts ---")

    def select(n):
        """select a leaf node in the tree"""
        if n.children:
            return select(max(n.children.keys(), key=ucb))
        else:
            return n

    def expand(n):
        """expand the leaf node by adding all its children states"""
        if not n.children and not gamestate.checkGameOver():
            n.children = {MCT_Node(state=gamestate.move(action), parent=n): action
                          for action in gamestate.actions()}
        return select(n)

    def simulate(gamestate):
        """simulate the utility of current state by random picking a step"""
        while not gamestate.checkGameOver():
            action = random.choice(list(gamestate.actions()))
            gamestate = gamestate.move(action)
        v = gamestate.getScore()
        return -v

    def backprop(n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        # if utility == 0:
        #     n.U += 0.5
        n.N += 1
        if n.parent:
            backprop(n.parent, -utility)

    root = MCT_Node(state=gamestate)

    #TOBE DELETED ----------
    #print("try mct_node state")
    #print(root.state.print())

    #TOBE DELETED ----------
    #print("")
    #print("starting board: ")
    #gamestate.print()
    #print("")

    for _ in range(N):
        #TOBE DELETED ----------
        #print("simulation number {}".format(str(_)))

        leaf = select(root)
        child = expand(leaf)
        result = simulate(gamestate)
        backprop(child, result)

    max_state = max(root.children, key=lambda p: p.N)

    return root.children.get(max_state)