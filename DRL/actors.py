import numpy as np
from time import sleep

from env import Env2048

from agent import Agent



class actor():
    def __init__(self, agent:Agent, action_ratio=1):
        self.agent = agent

        self.pred_state = None
        self.previous_state = None
        self.state = None
        self.action = None
        self.previous_action = self.action
        self.second_previous_action = self.previous_action
        self.reward = 0
        self.previous_reward = 0
        self.info = None
        self.previous_info = None
        self.q = None
        self.env = None

        self.world = None
        self.stage = None

        self.playing_game = True

        self.action_ratio = action_ratio
        self.action_furlough = 0
        self.action_steps = 0

    def act(self, maintain_sync=True, model='eval_model'):
        self.action, self.q, _, _ = self.agent.act(self.state, maintain_sync=maintain_sync, model=model)

    def env_step(self, env:Env2048):
        self.next_state, self.reward, self.done, self.info = env.step(self.action)
    
    def log_step(self):
        self.agent.logger.log_step(self.reward - self.previous_reward, *self.agent.learn_results())

    def cache_step(self):
        if self.info['history'] is not None:
            for mem in self.info['history']:
                for r in range(4):
                    self.agent.cache(
                            state=np.rot90(mem.next_state, r, (1,2)).copy(),
                            action=(mem.action-r)%self.agent.action_dim,
                            reward=mem.reward,
                            done=mem.done,
                        )

                    self.agent.cache(
                            state=np.rot90(np.fliplr(mem.next_state), r, (1,2)).copy(),
                            action=(-mem.action-r)%self.agent.action_dim,
                            reward=mem.reward,
                            done=mem.done,
                        )

    def log_episode(self):
        self.agent.logger.log_episode()

        if self.agent.sync_overseer.training:
            self.agent.logger.record(
                episode=self.agent.episode-1,
                epsilon=self.agent.exploration_rate*self.agent.exploration_modifier,
                learningrate=self.agent.optimizer_learningrate*self.agent.learningrate_modifier,
                memory=(len(self.agent.archive)),
                level=(self.world, self.stage),
                cur_level=0,
                step=self.agent.curr_step,
                save_dir=self.agent.save_dir,
            )

    def run(self):
        print("Run method not implimented for base actor class")




class actor_primary(actor):
    def __init__(self, agent: Agent):
        super().__init__(agent, 1)

        self.agent.sync_overseer.register_actor(self)

        self.wins = 0

        self.env = agent.env


    def run(self):
        assert self.action_ratio == 1

        self.next_state = self.env.reset()
        self.state = self.next_state

        ### for Loop that train the model num_episodes times by playing the game
        while self.playing_game:
            
            self.playing_episode = True

            # Play the game!
            while self.playing_episode:

                # self.sync_with_train()
                self.agent.sync_overseer.sync_actors(self)


                self.act()

                self.env_step(self.agent.env)

                # 7. Remember
                self.log_step()

                self.cache_step()

                self.agent.log_reward(self.reward)

                # 10. Update state and latent
                self.state = self.next_state
                self.previous_reward = self.reward


                # 6. Check if end of game
                if self.done:

                    self.next_state = self.agent.next_episode()
                    self.state = self.next_state

                    self.previous_reward = 0

                    self.playing_episode = False

            self.log_episode()

                

class actor_secondary(actor):
    def __init__(self, agent: Agent, action_ratio=1):
        super().__init__(agent, action_ratio)

        self.agent.sync_overseer.register_actor(self)

        self.env = Env2048(verbose=False, scoreHeuristic=agent.net_heuristic)

    def run(self):

        self.next_state = self.env.reset()
        self.state = self.next_state

        ### for Loop that train the model num_episodes times by playing the game
        while self.playing_game:
            
            self.playing_episode = True

            # Play the game!
            while self.playing_episode:

                # self.sync_with_train()
                self.agent.sync_overseer.sync_actors(self)

                self.act(maintain_sync=False)

                self.env_step(self.env)

                # 7. Remember
                # self.log_step()

                self.cache_step() # save experience

                # 10. Update state and latent
                self.state = self.next_state
                self.previous_reward = self.reward
                # previous_action = action


                # 6. Check if end of game
                if self.done:

                    self.next_state = self.env.reset()
                    self.state = self.next_state

                    self.previous_reward = 0

                    self.playing_episode = False

            # self.log_episode() #Secondary actor does not log


class actor_random(actor):
    def __init__(self, agent: Agent, action_ratio=1):
        super().__init__(agent, action_ratio)

        self.agent.sync_overseer.register_actor(self)

        self.env = Env2048(verbose=False, random_start=True, scoreHeuristic=agent.net_heuristic)

    def run(self):

        self.next_state = self.env.reset()
        self.state = self.next_state

        ### for Loop that train the model num_episodes times by playing the game
        while self.playing_game:
            
            self.playing_episode = True

            # Play the game!
            while self.playing_episode:

                # self.sync_with_train()
                self.agent.sync_overseer.sync_actors(self)

                self.act(maintain_sync=False)

                self.env_step(self.env)

                # 7. Remember
                # self.log_step()

                self.cache_step() # save experience

                # 10. Update state and latent
                self.state = self.next_state
                self.previous_reward = self.reward
                # previous_action = action


                # 6. Check if end of game
                if self.done:

                    self.next_state = self.env.reset()
                    self.state = self.next_state

                    self.previous_reward = 0

                    self.playing_episode = False

            # self.log_episode() #Secondary actor does not log


class bot():
    def __init__(self, agent:Agent):
        self.playing_game = True
        self.agent = agent


class bot_learn(bot):
    def run(self):
        while self.playing_game:
            # print("learn", playing_game)
            # 8. Learn
            sleep(0.01)
            mem = self.agent.learn()

            # # 9. Logging
            self.agent.last_loss, self.agent.last_AEnc_loss, self.agent.last_q = (
                np.log10(mem.loss) if mem.loss else None,
                # np.log10(mem.AEnc_loss) if mem.AEnc_loss else None,
                mem.AEnc_loss,
                mem.q,
            )


class bot_cycle(bot):
    def run(self):
        while self.playing_game:
            while not self.agent.begin_cycle:
                sleep(0.01)
            self.agent.sync_overseer.sync_cycle()
            self.agent.begin_cycle = False
            # print("cycle")
            self.agent.cycle()
            # print("cycle end")


class bot_recall(bot):
    def run(self):
        while self.playing_game:
            while not (len(self.agent.future_batches) < 1 and len(self.agent.archive) > self.agent.burnin+1):
                self.agent.sync_overseer.time_waiting["recall"] += 1
                sleep(0.01)
            # print("recall", len(self.agent.future_batches))
            self.agent.recall()
            # print("recall end")


class bot_save(bot):
    def run(self):
        while self.playing_game:
            # print("save", playing_game)
            while self.agent.save_dict is None:
                sleep(0.01)
            print("\tSaving...")
            self.agent.save()