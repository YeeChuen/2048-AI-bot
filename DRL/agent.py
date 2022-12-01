import os

import torch
import numpy as np

from env import Env2048

from net import DeepQLearner
from metrics import MetricLogger

from collections import deque
import copy
import math

from time import sleep

from tools import mem_state, mem_stack


class sync_overseer():
    def __init__(self, ratio):
        
        self.swing = 10 # how out of sync actors can be

        self.training = False

        self.train_cycle_steps = 0
        self.train_steps = 0
        self.cycle_steps = 0
        self.action_steps = 0

        self.learning_burnin = 0

        self.mean_loss = np.inf

        self.actor_steps = {}

        self.ratio = ratio

        self.time_waiting = {"train":0, "train-batch":0, "cycle":0, "recall":0, "actors":0}

    def register_actor(self, actor):
        self.actor_steps[actor] = 0

    def sync_actions(self):
        if self.training:
            # print('actions',self)
            self.action_steps += 1

    def sync_train(self):
        self.train_cycle_steps += 1
        if self.training:
            if self.learning_burnin < 0:
                self.train_steps += 1
        else:
            sleep(0.01)
            # print('train',self)
        while self.actor_steps and min(self.actor_steps.values()) < (self.train_steps - self.swing) // self.ratio:
            sleep(0.01)
            self.time_waiting["train"] += 1

    def sync_cycle(self):
        if self.training:
            self.cycle_steps += 1
            # print('cycle',self)
        while self.train_steps < self.cycle_steps - self.swing:
            sleep(0.01)
            self.time_waiting["cycle"] += 1

    def sync_actors(self, actor):
        # recovering = False
        if self.training:
            self.actor_steps[actor] += 1
            # print('actors',self)
        while (self.learning_burnin > 0 and self.training) or self.train_steps < int(self.actor_steps[actor] / actor.action_ratio * self.ratio) - self.swing:
            # print(6.3E-4 if recovering else 1E-3, recovering)
            sleep(0.01)
            # if self.mean_loss > 1E-3:
            #     recovering = True
            self.time_waiting["actors"] += 1

    def __repr__(self) -> str:
        return f"{self.__dict__}"


### ==================================================== INIT ==================================================== ###

class Agent:
    def __init__(self, input_dim, action_dim, latent_dim, save_dir, env:Env2048, checkpoint=None, cuda=False, save=True, testing=False):
        
        ### --------- Hyper Parameters --------- ###

        # self.deque_size = 1400
        
        self.archive_size = 2000000 #self.deque_size * 10
        self.batch_size = 200000
        self.encoder_batch_size = self.batch_size
        self.mini_batch_size = self.batch_size
        print("self.mini_batch_size", self.mini_batch_size)

        self.exploration_modifier_cycle = 8 # episodes
        self.exploration_modifier = 1
        self.exploration_modifier_range = [0.1, 1.0]
        self.exploration_rate = 0.0
        self.exploration_rate_decay = 0.9998
        self.exploration_rate_min = 0.0

        self.burnin = max(self.mini_batch_size*2, self.batch_size)  # min. frames before training
        self.update_steps_per_train = 1
        self.train_steps_per_frame = 1/8

        self.loss_hist = deque([], 10)

        self.initial_burnin = 50

        self.save_every = 3000   # no. of frames between saving Agent Net

        self.max_norm = 0.001
        self.optimizer_learningrate = 0.0005
        self.learningrate_cycle = 20 # frames
        self.learningrate_modifier = 1
        self.learningrate_modifier_range = [0.3, 0.7]


        ### --------- Initialization --------- ###

        if cuda and torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

        self.logger  = MetricLogger(save_dir)

        self.archive = np.array([], dtype=object).reshape((0,))
        self.archive_pointer = 0
        self.archive_indices = np.ones(shape=(0,), dtype=int)
        self.transfer_queue = []
        
        self.archive_lock = False
        self.transfer_queue_lock = False
        self.eval_lock = 0 # (-1: "update", 0: "neutral", 1: "evaluating")

        self.use_mcts = env.use_mcts

        self.input_dim, self.action_dim, self.latent_dim = input_dim, action_dim, latent_dim

        self.save_model=save
        self.env = env

        self.curr_step = 0
        self.episode = 0
        
        self.sync_overseer = sync_overseer(self.train_steps_per_frame)

        self.save_dir = save_dir
        self.save_dict = None
        self.testing = testing

        self.begin_cycle = False

        self.future_batches = []

        self.last_loss, self.last_AEnc_loss, self.last_q = None, None, None

        # Agent's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = DeepQLearner(input_dim, action_dim, latent_dim, self.batch_size).float()
        self.net = self._to_cuda(self.net)

        self.net.sync_eval()

        self.cpu_net = None

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=2E-4, amsgrad=True)
        # self.loss_fn = torch.nn.SmoothL1Loss()
        self.loss_fn = torch.nn.MSELoss()

        ### --------- Load Checkpoint --------- ###

        if checkpoint:
            self.load(checkpoint)
        else:
            self.sync_overseer.learning_burnin = self.initial_burnin

        self.last_sync = None
        self.last_save = self.curr_step
        self.last_save_path = None





    def reset(self):
        self.archive = np.array([], dtype=object).reshape((0,))
        self.future_batches = []

        self.env.reset()


    def next_episode(self):
        print("\n")

        if self.sync_overseer.training:
            self.episode += 1
        self.exploration_modifier = (math.cos(2*math.pi*self.episode/self.exploration_modifier_cycle)/2+0.5)*(self.exploration_modifier_range[1]-self.exploration_modifier_range[0])+self.exploration_modifier_range[0]
        next_state = self.env.reset()

        self.net.next_episode()
        self.net = self._to_cuda(self.net)

        print('Save in:', self.save_every - (self.curr_step - self.last_save), 'frame(s)')
        print('Memory:', len(self.archive), len(self.transfer_queue))

        return next_state

    
    def learn_results(self):
        if self.last_loss is not None:
            self.eval_lock = True
            sleep(0.01)
            self.net.sync_eval()
            self.eval_lock = False
            
        results = self.last_loss, self.last_AEnc_loss, self.last_q
        self.last_loss, self.last_AEnc_loss, self.last_q = None, None, None
        return results


    def _to_cuda(self, tensor):
        if self.cuda:
            return tensor.cuda()
        else:
            return tensor.cpu()


### ==================================================== ACT ==================================================== ###


    def net_heuristic(self, state2048):
        while self.eval_lock:
                sleep(0.01)
        action_value = self.net(self._to_cuda(torch.FloatTensor(state2048.oneHot())).unsqueeze(0), model='eval_model').detach().cpu()
        action_value = torch.add(action_value.squeeze(), torch.tensor(state2048.getValidMoves(), requires_grad=False))
        return action_value.numpy()


    def act(self, env, maintain_sync=True, model='eval_model'):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Agent will perform
        """

        # EXPLOIT

        state = self._to_cuda(torch.FloatTensor(env.previous_state))
        state = state.unsqueeze(0)
        
        if model == "eval_model":
            while self.eval_lock:
                sleep(0.01)
        action_value = self.net(state, model=model).detach().cpu()
        action_value = torch.add(action_value.squeeze(), torch.tensor(env.valid_moves, requires_grad=False))


        # EXPLORE
        if np.random.rand() < self.exploration_rate * self.exploration_modifier or len(self.archive) < self.burnin:
            action_idx = np.random.choice([i for i in range(self.action_dim) if env.valid_moves[i]>-np.inf])
        else:
            if env.use_mcts:
                action_idx = env.mcts_simulate()
            if not env.use_mcts or env.valid_moves[action_idx] == -np.inf:
                action_idx = torch.argmax(action_value)

        
        action_value = action_value[action_idx]
        
        if maintain_sync:
            # decrease exploration_rate
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

            self.sync_overseer.sync_actions()
            if self.sync_overseer.training:
                self.curr_step += 1

        assert env.valid_moves[action_idx] > -np.inf and action_value > -np.inf

        return action_idx, action_value, None, None


    


### ==================================================== LEARN ==================================================== ###



    @torch.no_grad()
    def td_target(self, mem:mem_state):
        ### target rewards

        td_target=mem.batch.reward/self.env.running_max

        return mem.merge(
            td_target=td_target

        )


    def update_online(self, mem:mem_state):
        loss = []
        AEnc_loss = []
        # print(1)
        td_estimate = None

        batch_idx = np.random.shuffle(np.arange(0, len(mem.batch.state)))

        for i in range(math.ceil(len(mem.batch.state)/self.mini_batch_size)):

            x = np.arange(i*self.mini_batch_size, min(len(mem.batch.state), (i+1)*self.mini_batch_size))

            if len(x) < self.mini_batch_size and td_estimate is not None:
                continue
            
            _x = x

            for _ in range(self.update_steps_per_train):
                current_Q = self.net(mem.batch.state[_x], model='online') # Q_online(s,a)
                _td_estimate = current_Q[np.arange(0, len(current_Q)), mem.batch.action[_x]]

                mb_loss = self.loss_fn(_td_estimate, mem.td_target[_x])

                if td_estimate is None:
                    td_estimate = _td_estimate
                else:
                    td_estimate = torch.concat([td_estimate, _td_estimate])

                self.optimizer.zero_grad()
                mb_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
                self.optimizer.step()

                loss.append(mb_loss.item())

        return mem.merge(
            loss = mem.loss + np.mean(loss),
            AEnc_loss = None,#np.mean(AEnc_loss),
            td_estimate = td_estimate,
        ) 


    def learn(self):
        self.sync_overseer.sync_train()
        self.begin_cycle = True

        mem = mem_state(q=None, loss=None, AEnc_loss=None)

        if self.save_model and self.curr_step > self.last_save + self.save_every:
            self.begin_save()
            self.last_save = self.curr_step
            
        if len(self.archive) > self.burnin+1:
            self.sync_overseer.training = True

            self.learningrate_modifier = (math.cos(2*math.pi*self.sync_overseer.train_cycle_steps/self.learningrate_cycle)/2+0.5)*(self.learningrate_modifier_range[1]-self.learningrate_modifier_range[0])+self.learningrate_modifier_range[0]
            for g in self.optimizer.param_groups:
                g['lr'] = self.optimizer_learningrate * self.learningrate_modifier

            ### Sample from memory
            while len(self.future_batches)==0:
                self.sync_overseer.time_waiting["train-batch"] += 1
                sleep(0.01)

            mem = self.future_batches.pop(0)
            mem = mem.merge(loss=0, AEnc_loss=None)

            ### Train the network
            mem = self.td_target(mem)
            mem = self.update_online(mem)

            ### Cleanup batch
            del mem.batch
            
            ### log and finish batch
            mem.q = mem.td_estimate.mean().item()

            self.loss_hist.append(mem.loss)
            self.sync_overseer.mean_loss = np.mean(self.loss_hist)
            self.sync_overseer.learning_burnin -= 1
            if self.sync_overseer.learning_burnin > 0:
                print("burnin:", self.sync_overseer.learning_burnin, np.log10(self.sync_overseer.mean_loss), np.std(np.log10(self.loss_hist)))


        return mem


### ==================================================== CACHE/RECAL ==================================================== ###

    def cache(self, **args):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        TODO: update
        """

        mem = mem_state(
            state = (torch.ByteTensor(args["state"])),
            action = (torch.LongTensor([args["action"]])),
            reward = (torch.FloatTensor([args["reward"]])),
            done = (torch.BoolTensor([args["done"]])),
        )

        self.transfer_queue.append(mem)


    def cycle(self):
        ### cycle new samples into archive

        if len(self.transfer_queue):
            transfer, self.transfer_queue = self.transfer_queue, []

        else:
            transfer = []
            return

        while self.archive_lock == True:
            sleep(0.01)
            
        self.archive_lock = True

        if len(transfer):
            self.archive = np.concatenate([list(reversed(transfer)), self.archive])
        self.archive = self.archive[:self.archive_size]

        self.archive_lock = False

        
    def log_reward(self, reward, model='online'):
        pass


    def recall(self):
        """
        Retrieve a batch of experiences from memory 
        """
        
        while self.archive_lock == True:
            sleep(0.01)
            
        self.archive_lock = True

        batch = np.random.choice(self.archive, self.batch_size)

        self.archive_lock = False

        ### stack samples and move batch to gpu
        mem = mem_stack(batch)
        vars = mem.vars()
        for k in vars:
            vars[k] = self._to_cuda(torch.stack(vars[k]).squeeze())
            
        self.future_batches.append(mem_state(batch=mem))



### ==================================================== SAVE/LOAD ==================================================== ###

    def begin_save(self):
        self.save_dict = dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                curr_step=self.curr_step,
                episode=self.episode,
                archive=self.archive.copy(),
                logger=copy.deepcopy(self.logger),
            )


    def save(self):
        save_path = self.save_dir / f"_mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        
        _save_dict, self.save_dict = self.save_dict, None

        torch.save(
            _save_dict,
            save_path
        )

        try:
            if self.last_save_path and os.path.isfile(self.last_save_path):
                os.remove(self.last_save_path)
        except:
            pass

        self.last_save_path = save_path

        print(f"\tAgentNet saved to {save_path} at step {self.curr_step}")

        del _save_dict

        if int(self.curr_step // self.save_every) >= 27: exit()


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=("cpu"))

        exploration_rate = ckp.get('exploration_rate')
        self.curr_step = ckp.get('curr_step')
        self.episode = ckp.get('episode')

        state_dict = ckp.get('model')
        self.current_level = ckp.get('current_level')

        self.logger = ckp.get('logger')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate} at {self.curr_step}")
        self.net.load_state_dict(state_dict)
        
        if self.cuda:
            self.net = self.net.to(device="cuda")
        else:
            self.net = self.net.to(device="cpu")

        self.exploration_rate = exploration_rate

        self.archive = ckp.get('archive')

        for _mem in self.archive:
            for k in _mem:
                _mem[k] = _mem[k].cpu()

        self.archive = np.array(self.archive, dtype=object)
