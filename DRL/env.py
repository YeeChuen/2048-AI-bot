import gym
from tqdm import tqdm

import numpy as np
import random

from tools import mem_state

import matplotlib.pyplot as plt

from game_2048 import State2048

from MonteCarlo import MCTS


class Env2048(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose=True, random_start=False, scoreHeuristic=None):
        super(Env2048, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.scoreHeuristic = scoreHeuristic

        self.actions = [
            ['up'],
            ['right'],
            ['down'],
            ['left'],
        ]

        self.running_max = 2048

        self.use_mcts = True
        self.mcts_depth = 4
        self.mcts_sims = 3

        self.mcts = None

        self.verbose = verbose

        self.action_dim = len(self.actions)

        self.random_start = random_start

        self.frame = None
        self.history = None
        self.board = None
        self.previous_state = None

        self.valid_moves = None

        self.pbar = None
        if self.verbose:
            self.pbar = tqdm(total=0)
        
        self.reset()


    
    def report(self, *args):
        if self.verbose:
            if self.pbar is not None:
                self.pbar.write(' '.join(tuple(map(str,args))))
            else:
                print(*args)


    def reset(self):
        board = None
        if self.random_start:
            board = np.random.binomial(1, 0.3, (4,4)) * 2**np.int32(np.random.uniform(1, 11, (4,4)))

        self.board = State2048(board, scoreHeuristic=self.scoreHeuristic)

        if self.use_mcts == True:
            self.mcts = MCTS(self.board, self.mcts_sims, self.mcts_depth)

        if self.pbar is not None:
            self.pbar.reset(total = 0)

        self.frame = 0
        self.previous_state = self.board.oneHot()

        self.history = []

        self.valid_moves = self.board.getValidMoves()

        return self
        
    def mcts_simulate(self):
        action_values = self.mcts.simulation()
        # print(action_values, self.valid_moves)
        action_values = np.add(action_values, self.valid_moves)
        # if np.max(action_values) == -np.inf:
        #     return np.argmax(self.valid_moves + np.random.uniform(0,1,len(self.valid_moves)))
        # else:
        return np.argmax(action_values)

    def step(self, action):
        self.frame += 1
        
        if self.use_mcts:
            self.mcts.update_currnode(action, self.board)
            next_board = self.mcts.currstate
        else:
            next_board = self.board.move(action)

        if next_board is None:
            raise KeyError(f"action {action} is not valid")

        self.board = next_board

        next_state = next_board.oneHot()
        # reward = np.log2(next_board.h2())
        reward = next_board.h2()
        # print(reward)
        self.valid_moves = next_board.getValidMoves()
        done = not (np.max(self.valid_moves) > -np.inf)

        self.history.append(mem_state(state=self.previous_state, next_state=next_state, reward=reward, done=done, action=action))

        self.previous_state = next_state

        info = dict(history=None)
        if done:
            # self.running_max = max(reward, self.running_max)
            # reward /= self.running_max

            for mem in self.history:
                mem.reward = reward - mem.reward

            print(self.board.score, len(self.history))
            
            info['history'] = self.history

            self.report(next_board)

        if self.pbar is not None:
            self.pbar.update(1)

        return self, reward, done, info

    
