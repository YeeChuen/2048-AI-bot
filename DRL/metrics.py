import numpy as np
import time, datetime
import matplotlib.pyplot as plt

from tools import mem_state

class MetricLogger():
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.reset()


    def log_step(self, reward, loss, AEnc_loss, q):
        self.curr_ep.reward.append(reward)
        self.curr_ep.length += 1
        if loss is not None:
            self.curr_ep.loss.append(loss)
            self.curr_ep.q.append(q)

        if AEnc_loss is not None:
            self.curr_ep.AEnc_loss.append(AEnc_loss)
            

    def log_episode(self):
        "Mark end of episode"
        self.history.reward.append(sum(self.curr_ep.reward))
        self.history.length.append(self.curr_ep.length)

        if len(self.curr_ep.loss) == 0:
            self.history.loss.append(0)
            self.history.AEnc_loss.append(0)
            self.history.q.append(0)

            self.history.sd_loss.append(0)
            self.history.sd_AEnc_loss.append(0)
            self.history.sd_q.append(0)

        else:
            self.history.loss.append( np.mean(self.curr_ep.loss) )
            # self.history.AEnc_loss.append( np.mean(self.curr_ep.AEnc_loss) )
            self.history.q.append( np.mean(self.curr_ep.q) )

            self.history.sd_loss.append( np.std(self.curr_ep.loss) )
            # self.history.sd_AEnc_loss.append( np.std(self.curr_ep.AEnc_loss) )
            self.history.sd_q.append( np.std(self.curr_ep.q) )

        self.init_episode()

    def init_episode(self):
        self.curr_ep = mem_state(
            reward=[],
            length=0,
            loss=[],
            AEnc_loss=[],
            q=[],
        )
        

    def reset(self):
        # History metrics
        self.history = mem_state(
            reward=[],
            length=[],
            loss=[],
            AEnc_loss=[],
            q=[],
            sd_loss=[],
            sd_AEnc_loss=[],
            sd_q=[],
        )

        # Current episode metric
        self.init_episode()

        self.record_time = time.time()

    def reset_time(self):
        self.record_time = time.time()

    def record(self, episode, epsilon, learningrate, memory, level, cur_level, step, save_dir):

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 4)

        self.save_dir = save_dir

        print(
            f"Episode {episode} - ",
            f"Frame {step} - ",
            f"Epsilon {round(epsilon,3)} - ",
            f"log_LR {round(np.log10(learningrate),3)} - ",
            f"Mean Reward {self.history.reward[-1]} - ",
            f"Mean Length {self.history.length[-1]} - ",
            f"Mean Loss {(self.history.loss[-1], self.history.sd_loss[-1])} - ",
            # f"Mean AEnc_Loss {(self.history.AEnc_loss[-1], self.history.sd_AEnc_loss[-1])} - ",
            f"Mean Q Value {(self.history.q[-1], self.history.sd_q[-1])} - ",
            f"Final Level {level} - ",
            f"Max Level {cur_level+1} - ",
            f"Memory {memory} - ",
            f"\nTime Delta {time_since_last_record} - ",
            f"Time Per Frame {time_since_last_record/self.history.length[-1]} - ",
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        )


        plt.clf()

        for metric in ["reward", "length"]:
            y = [x for i,x in enumerate(self.history[metric]) if self.history["q"][i]!=0]
            print(metric, np.mean(y[-10:]))
            if len(y)>3:
                plt.plot(y[:-2])
                plt.plot([len(y)-3,len(y)-2],y[-3:-1], color = 'magenta')
                plt.plot([len(y)-2,len(y)-1],y[-2:], color = 'red')
            else:
                plt.plot(y)
            plt.savefig(self.save_dir / f"{metric}_plot")
            plt.clf()

        metric = "q"
        y = np.array([x for i,x in enumerate(self.history[metric]) if self.history["q"][i]!=0])
        print(metric, np.mean(y[-10:]))
        z = np.array([x for i,x in enumerate(self.history[f"sd_{metric}"]) if self.history["q"][i]!=0])
        plt.fill_between(list(range(len(y))),y-z, y+z, alpha=0.4)
        if len(y)>3:
            plt.plot(y[:-2])
            plt.plot([len(y)-3,len(y)-2],y[-3:-1], color = 'magenta')
            plt.plot([len(y)-2,len(y)-1],y[-2:], color = 'red')
        else:
            plt.plot(y)
        plt.savefig(self.save_dir / f"{metric}_plot")
        plt.clf()


        # for metric in ["loss", "AEnc_loss"]:
        for metric in ["loss"]:
            y = np.array([x for i,x in enumerate(self.history[metric]) if self.history["q"][i]!=0])
            print(metric, np.mean(y[-10:]))
            z = np.array([x for i,x in enumerate(self.history[f"sd_{metric}"]) if self.history["q"][i]!=0])
            plt.fill_between(list(range(len(y))),y-z, y+z, alpha=0.4)
            if len(y)>3:
                plt.plot(y[:-2], label=metric)
                plt.plot([len(y)-3,len(y)-2],y[-3:-1], color = 'magenta')
                plt.plot([len(y)-2,len(y)-1],y[-2:], color = 'red')
            else:
                plt.plot(y)
        # plt.legend()   
        plt.savefig(self.save_dir / f"{metric}_plot")
        plt.clf()
