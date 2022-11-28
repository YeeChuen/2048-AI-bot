# Author: Yee Chuen Teoh
# Project: 572 Term Project
# Title: 
# Description: 
# Reference: https://github.com/pandezhao/alpha_sigma

# ______________________________________________________________________________
# Imports
import numpy as np
import utils
import sys
import time
import random

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
    def __init__(self, parent = None, state2048 = None, depth = 0, priorAction = None):
        self.parent = parent    # parent is a node
        self.state2048 = state2048
        self.hashvalue = state2048.__hash__()   # hashvalue to name the node <node hashvalue>
        self.score = state2048.score #basically h1(heuristic 1)
        self.high_score_action = [0,0,0,0]
        self.child_from_action = [0,0,0,0]
        #self.simulation_count = 0
        self.depth = depth      # depth to limit the search depth (speed up search)
        self.child = {}         # child uses set to prevent duplicate
        self.priorAction = priorAction # what action leads to this node?

    # backpropagation should return the parent node with updated score
    # the high score action should be based on average score for that move. 
    def backprop(self):
        if self.parent: #check if the node has parent
            if sum(self.high_score_action) != 0:    #check if current node already has a highscore from child node
                                                        #if yes, use that to update parent node
                #TOBE DELETED
                #print("sum satisfied in backprop")

                if self.parent.high_score_action[self.priorAction] != 0:
                    #TOBE DELETED
                    #print("parent in that action has existing")

                    # below is code for average
                    prev_total = self.parent.child_from_action[self.priorAction]
                    score_total = self.parent.high_score_action[self.priorAction]*(prev_total-1) + max(self.high_score_action)
                    self.parent.high_score_action[self.priorAction] =score_total/prev_total
                    
                    # below is code for max
                    self.parent.high_score_action[self.priorAction] = max(max(self.high_score_action), self.parent.high_score_action[self.priorAction])
                else:
                    #TOBE DELETED
                    #print("parent in that action empty")

                    self.parent.high_score_action[self.priorAction] = max(self.high_score_action)
            else:
                #TOBE DELETED
                #print("do not have previous child score")

                if self.parent.high_score_action[self.priorAction] != 0:
                    # below is code for average
                    prev_total = self.parent.child_from_action[self.priorAction]
                    score_total = self.parent.high_score_action[self.priorAction]*(prev_total-1) + self.score
                    self.parent.high_score_action[self.priorAction] =score_total/prev_total
                    
                    # below is code for average
                    #self.parent.high_score_action[self.priorAction] = max(max(self.high_score_action), self.parent.high_score_action[self.priorAction])
                
                else:
                    self.parent.high_score_action[self.priorAction] = self.score
            return self.parent
        return None
        '''
        if self.parent: #check if the node has parent
            if self.high_score_action:  #check if current node already has a highscore from child node
                                        #if yes, use that to update parent node
                self_keylist = list(self.high_score_action.keys())
                self_key=self_keylist[0]
                if self.parent.high_score_action:   #check if the parent already has other highscore
                    keylist = list(self.parent.high_score_action.keys())
                    key=keylist[0]
                    if self_key > key:
                        del self.parent.high_score_action[key]
                        self.parent.high_score_action[self_key]=self.priorAction
                else:
                        self.parent.high_score_action[self_key]=self.priorAction
            else:
                if self.parent.high_score_action:
                    keylist = list(self.parent.high_score_action.keys())
                    key=keylist[0]
                    if self.score > key:
                        del self.parent.high_score_action[key]
                        self.parent.high_score_action[self.score]=self.priorAction
                else:
                    self.parent.high_score_action[self.score]=self.priorAction
            return self.parent
        return None
        '''

    def count_child(self):
        for key in self.child:
            print("Number of Child in from action {}: {}".format(str(key), str(len(self.child[key]))))

    def exists(self, node, action):
        for childnode in self.child[action]:
            if node.hashvalue == childnode.hashvalue:
                return True
        return False

    def get_samechild(self, node, action):
        for childnode in self.child[action]:
            if node.hashvalue == childnode.hashvalue:
                return childnode
        print("that childnode does not exist")
        return None

    # addchild 
    def addchild(self, action):
        self.child_from_action[action]+=1
        childstate = self.state2048.move(action)
        childnode = node(parent = self, state2048=childstate, depth=self.depth+1, priorAction=action)
        # check the key(action) for child exists
        if action in self.child:
            # check if there is this same child?
            if self.exists(childnode, action) is True:
                return self.get_samechild(childnode, action)
            else:
                self.child[action].add(childnode)
        else:
            self.child[action]={childnode}
        return childnode

    # print node information
    def print(self):  
        print("-----------------node information------------------")
        print("node: {}".format(self.__repr__()))
        print("prior action: {}".format(self.priorAction))
        print("parent node: {}".format(str(self.parent)))
        print("child nodes: {}".format(str(self.child)))
        print("node depth: {}".format(str(self.depth)))
        print("node score: {}".format(str(self.score)))
        print("node possible highest score: {}".format(str(self.high_score_action)))
        print("node child from action: {}".format(str(self.child_from_action)))
        self.count_child()
        #print("node simulation count: {}".format(str(self.simulation_count)))
        print("node state:")
        if self.state2048:
            self.state2048.print() 
        print("---------------------------------------------------")

    # TODO how to change this up since we calculate utility by score
    # this function is currently irrelevant
    def ucb(n, C=1.4):
        return np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N)

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self.hashvalue)

# ______________________________________________________________________________
# Monte Carlo Search
class MCTS:
    def __init__(self, state2048, no_simulation=100, depth = 50):
        self.node_state = state2048 # this could be redundant
        #self.maxscore = 0       # this could be redundant
        self.no_simulation = no_simulation
        self.search_depth = depth
        self.currnode = node(state2048=self.node_state)

    def print(self):
        print("-----------------MCTS information------------------")
        print("MCTS number of simulation: {}".format(str(self.no_simulation)))
        print("MCTS search depth: {}".format(str(self.search_depth)))
        print("---------------------------------------------------")

    def update_currnode(self, action):
        self.search_depth+=1
        self.currnode = self.currnode.addchild(action)

    def greedy_action(self, node, actionlist):
        state = node.state2048
        action_score = [0,0,0,0]
        for action in actionlist:
            newstate = state.move(action)
            action_score[action] = newstate.score #score is h1 (heuristic 1)
        # if all action the same, choose the a random action
        if sum(action_score) == 0:
            return np.random.choice(actionlist)
        return max_index(action_score)

    def simulation(self):
        currnode = self.currnode
        for _ in range(self.no_simulation):
            '''
            1. check all available action of the node, randomly pick one
            2. expand the action until a depth limit is reached, then stop the expansion
            3. that would be 1 simulation
            4. for n+1 simulation, check if the same node with the same board already existed as a child,
            5. if not, create a new child
            '''
            while currnode.depth < self.search_depth:
                action = -1
                action_list = currnode.state2048.actions()
                if not action_list:
                    break

                #for first child always choose greedy option
                if sum(currnode.child_from_action) == 0:
                    action = self.greedy_action(currnode, action_list)
                #after first child, 50/50 greedy or not
                else:
                #choose greedy 
                    if random.random() < .2:
                        action = self.greedy_action(currnode, action_list)
                #not greedy, choose at random
                    else:
                        action =np.random.choice(action_list)

                childnode = currnode.addchild(action)

                currnode = childnode
            
            # back prop conditional here always bring node back to the parent
            # change this to back to current node
            while currnode is not self.currnode:
                #TOBE DELETED
                #currnode.print()

                currnode = currnode.backprop()   
            #TOBE DELETED  
            #currnode.print()
        return max_index(self.currnode.high_score_action)
        