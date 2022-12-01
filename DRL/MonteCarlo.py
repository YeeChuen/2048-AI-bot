# Author: Yee Chuen Teoh
# Project: 572 Term Project
# Title: 
# Description: 
# Reference: https://github.com/pandezhao/alpha_sigma

# ______________________________________________________________________________
# Imports
import numpy as np
import sys


def max_index(list):
    maxnum = max(list)
    list_idx = []
    for x in range(len(list)):
        if list[x] == maxnum:
            list_idx.append(x)
    return np.random.choice(list_idx)
# ______________________________________________________________________________
# MCT node
class node:
    
    # possibly add a depth limitation here
    def __init__(self, parent = None, depth = 0, priorAction = None):
        self.parent = parent    # parent is a node
        self.U = [0]*4
        self.N = [0]*4
        #self.simulation_count = 0
        self.depth = depth      # depth to limit the search depth (speed up search)
        self.child = {}         # child uses set to prevent duplicate
        self.priorAction = priorAction # what action leads to this node?

    # backpropagation should return the parent node with updated score
    # the high score action should be based on average score for that move. 
    def backprop(self):
        if self.parent: #check if the node has parent
            if max(self.N) > 0:

                self.parent.N[self.priorAction] += np.sum(self.N)

                self.parent.U[self.priorAction] += np.sum(self.U)
            return self.parent
        return None

    # addchild 
    def simaction(self, action, state2048):
        nextstate = state2048.move(action)
        # check the key(action) for child exists
        valid = nextstate.getValidMoves()
        if action not in self.child:
            childnode = node(parent = self, depth=self.depth+1, priorAction=action)
            self.child[action] = childnode
        else:
            childnode = self.child[action]

        h = nextstate.h
        for a in range(4):
            if valid[a] > -np.inf:
                childnode.N[a]+=1
                childnode.U[a]+=h[a]
        # self.backprop()
        return childnode, nextstate

    
    # def ucb(n, C=1.4):
    #     i = np.argmax(np.add(n.U, [np.inf if n.N[j]==0 else 0 for j in range(4)]))
    #     U=n.U[i]
    #     N=n.N[i]
    #     return sys.maxsize if N == 0 else U / N + C * np.sqrt(np.log(np.sum(n.parent.N)) / np.sum(n.N))

    def ucb(n, C=1.4):
        i = np.argmax(np.add(n.U, [np.inf if n.N[j]==0 else 0 for j in range(4)]))
        U=n.U[i]
        N=n.N[i]
        return sys.maxsize if N == 0 else U / N + C * np.sqrt(np.log(np.sum(n.parent.N)+1) / np.sum(n.N))


# ______________________________________________________________________________
# Monte Carlo Search
class MCTS:
    def __init__(self, state2048, no_simulation=100, depth = 50):
        self.currstate = state2048 

        self.no_simulation = no_simulation
        self.search_depth = depth
        self.currnode = node()

    def update_currnode(self, action, state2048):
        self.search_depth+=1
        self.currnode, self.currstate = self.currnode.simaction(action, state2048)

    def greedy_action(self, node, state2048, C=None):
        valid = state2048.getValidMoves()
        if not node.child:
            vals =  np.add(np.random.uniform(0,1,4), valid)
        elif C is None:
            vals = np.add([node.child[i].ucb() if i in node.child else sys.maxsize for i in range(4)], valid)
        else:
            vals = np.add([node.child[i].ucb(C=C) if i in node.child else sys.maxsize for i in range(4)], valid)

        j = np.argmax(vals)
        assert j in range(4)
        # print(j, (vals[j], valid[j]))
        return j, min(vals[j], valid[j])

    def simulation(self):
        # self.currnode = currnode = node(depth=self.currnode.depth)
        currnode = self.currnode
        currstate = self.currstate
        for _ in range(self.no_simulation):
            '''
            1. check all available action of the node, randomly pick one
            2. expand the action until a depth limit is reached, then stop the expansion
            3. that would be 1 simulation
            4. for n+1 simulation, check if the same node with the same board already existed as a child,
            5. if not, create a new child
            '''
            while currnode.depth < self.search_depth:

                action, v = self.greedy_action(currnode, currstate)

                if np.isinf(v):
                    break

                currnode, currstate = currnode.simaction(action, currstate)
            
            # back prop conditional here always bring node back to the parent
            # change this to back to current node
            while currnode is not self.currnode:
                currnode = currnode.backprop()

            currnode = self.currnode # redundant
            currstate = self.currstate
            
        return self.greedy_action(self.currnode, self.currstate, C=0)[0]
        # return np.clip(self.currnode.high_score_action, np.nanmin([np.nanmin(self.currnode.high_score_action), 0]), np.inf)
        


# # ______________________________________________________________________________
# # reference code starts here 
# # code below here is irrelevant

# num2char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m",
#             13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u"}

# distrib_calculater = utils.distribution_calculater(utils.board_size)

# # ______________________________________________________________________________
# # Edge
# class edge1:
#     def __init__(self, action, parent_node, priorP):
#         self.action = action
#         self.counter = 1.0
#         self.parent_node = parent_node
#         self.priorP = priorP    #TODO whats priorP?
#         self.child_node = None # self.search_and_get_child_node()

#         self.action_value = 0.0

#     def backup(self, v):  # back propagation
#         self.action_value += v
#         self.counter += 1
#         self.parent_node.backup(-v)

#     def get_child(self):
#         if self.child_node is None:
#             self.counter += 1
#             self.child_node = node(self, -self.parent_node.node_player)
#             return self.child_node, True
#         else:
#             self.counter += 1
#             return self.child_node, False

#     def UCB_value(self):  # 计算当前的UCB value
#         if self.action_value:
#             Q = self.action_value / self.counter
#         else:
#             Q = 0
#         return Q + utils.Cpuct * self.priorP * np.sqrt(self.parent_node.counter) / (1 + self.counter)

# # ______________________________________________________________________________
# # Node
# class node1:
#     def __init__(self, parent, player):
#         self.parent = parent
#         self.counter = 0.0
#         self.child = {}
#         self.node_player = player

#     def add_child(self, action, priorP):  # 增加node治下的一个edge，但是没有实际创建新的node
#         action_name = utils.move_to_str(action)
#         self.child[action_name] = edge1(action=action, parent_node=self, priorP=priorP)

#     def get_child(self, action):
#         child_node, _ = self.child[action].get_child()
#         return child_node

#     def eval_or_not(self):
#         return len(self.child)==0

#     def backup(self, v):  # back propagation
#         self.counter += 1
#         if self.parent:
#             self.parent.backup(v)

#     def get_distribution(self, train=True): ## used to get the step distribution of current
#         for key in self.child.keys():
#             distrib_calculater.push(key, self.child[key].counter)
#         return distrib_calculater.get(train=train)


#     def UCB_sim(self):  # 用于根据UCB公式选取node
#         UCB_max = -sys.maxsize
#         UCB_max_key = None
#         for key in self.child.keys():
#             if self.child[key].UCB_value() > UCB_max:
#                 UCB_max_key = key
#                 UCB_max = self.child[key].UCB_value()
#         this_node, expand = self.child[UCB_max_key].get_child()
#         return this_node, expand, self.child[UCB_max_key].action

# # ______________________________________________________________________________
# # Monte Carlo Tree Search
# class MCTS1:
#     def __init__(self, board_size=11, simulation_per_step=400, neural_network=None):
#         self.board_size = board_size
#         self.s_per_step = simulation_per_step
#         # self.database = {0: {"":node(init_node, 1, self)}}  # here we haven't complete a whole database that can be
#         # self.current_node = self.database[0][""]            # used to search the exist node
#         self.current_node = node(None, 1)
#         self.NN = neural_network
#         self.game_process = five_stone_game(board_size=board_size)  # 这里附加主游戏进程
#         self.simulate_game = five_stone_game(board_size=board_size)  # 这里附加用于模拟的游戏进程

#         self.distribution_calculater = utils.distribution_calculater(self.board_size)

#     def renew(self):
#         self.current_node = node(None, 1)
#         self.game_process.renew()

#     def MCTS_step(self, action):
#         next_node = self.current_node.get_child(action)
#         next_node.parent = None
#         return next_node

#     def simulation(self):  # simulation的程序
#         eval_counter, step_per_simulate = 0, 0
#         for _ in range(self.s_per_step):
#             expand, game_continue = False, True
#             this_node = self.current_node
#             self.simulate_game.simulate_reset(self.game_process.current_board_state(True))
#             state = self.simulate_game.current_board_state()
#             while game_continue and not expand:
#                 if this_node.eval_or_not():
#                     state_prob, _ = self.NN.eval(utils.transfer_to_input(state, self.simulate_game.which_player(), self.board_size))
#                     valid_move = utils.valid_move(state)
#                     eval_counter += 1
#                     for move in valid_move:
#                         this_node.add_child(action=move, priorP=state_prob[0, move[0]*self.board_size + move[1]])

#                 this_node, expand, action = this_node.UCB_sim()
#                 game_continue, state = self.simulate_game.step(action)
#                 step_per_simulate += 1

#             if not game_continue:
#                 this_node.backup(1)
#             elif expand:
#                 _, state_v = self.NN.eval(utils.transfer_to_input(state, self.simulate_game.which_player(), self.board_size))
#                 this_node.backup(state_v)
#         return eval_counter / self.s_per_step, step_per_simulate / self.s_per_step

#     def game(self, train=True):  # 主程序
#         game_continue = True
#         game_record = []
#         begin_time = int(time.time())
#         step = 1
#         total_eval = 0
#         total_step = 0
#         while game_continue:
#             begin_time1 = int(time.time())
#             avg_eval, avg_s_per_step = self.simulation()
#             action, distribution = self.current_node.get_distribution(train=train)
#             game_continue, state = self.game_process.step(utils.str_to_move(action))
#             self.current_node = self.MCTS_step(action)
#             game_record.append({"distribution": distribution, "action":action})
#             end_time1 = int(time.time())
#             print("step:{},cost:{}s, total time:{}:{} Avg eval:{}, Aver step:{}".format(step, end_time1-begin_time1, int((end_time1 - begin_time)/60),
#                                                     (end_time1 - begin_time) % 60, avg_eval, avg_s_per_step), end="\r")
#             total_eval += avg_eval
#             total_step += avg_s_per_step
#             step += 1
#         self.renew()
#         end_time = int(time.time())
#         min = int((end_time - begin_time)/60)
#         second = (end_time - begin_time) % 60
#         print("In last game, we cost {}:{}".format(min, second), end="\n")
#         return game_record, total_eval/step, total_step/step

#     def interact_game_init(self):
#         self.renew()
#         _, _ = self.simulation()
#         action, distribution = self.current_node.get_distribution(train=False)
#         game_continue, state = self.game_process.step(utils.str_to_move(action))
#         self.current_node = self.MCTS_step(action)
#         return state, game_continue

#     def interact_game1(self, action):
#         game_continue, state = self.game_process.step(action)
#         return state, game_continue

#     def interact_game2(self, action, game_continue, state):
#         self.current_node = self.MCTS_step(utils.move_to_str(action))
#         if not game_continue:
#             pass
#         else:
#             _, _ = self.simulation()
#             action, distribution = self.current_node.get_distribution(train=False)
#             game_continue, state = self.game_process.step(utils.str_to_move(action))
#             self.current_node = self.MCTS_step(action)
#         return state, game_continue