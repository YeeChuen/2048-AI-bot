if __name__ == '__main__':

    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    import datetime
    from pathlib import Path

    from env import Env2048

    from agent import Agent
    from metrics import MetricLogger

    import os
    import shutil
    from threading import Thread

    import random, time

    from actors import actor_primary, actor_secondary, actor_random, bot_cycle, bot_learn, bot_recall, bot_save


    env = Env2048()


    ### Save paths
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    checkpoint = None

    ### Example of loading checkpoint
    # save_dir = Path('checkpoints') / '2022-11-26T22-31-39'
    # checkpoint = save_dir.joinpath("_mario_net_22.chkpt")

    


    ### Initialize agent (controls learning and almost everything else)
    agent = Agent(
        input_dim=(16, 4, 4),
        action_dim=env.action_dim,
        latent_dim=(16, 4, 4),
        save_dir=save_dir,
        checkpoint=checkpoint,
        cuda=True,
        env=env,
    )

    env.scoreHeuristic = agent.net_heuristic
    env.reset()

    num_actors = 6
    num_random_actors = 0

    print("Episode:", agent.episode, "Frame:", agent.curr_step)

    ### Make experiment dir
    if not os.path.isdir(save_dir):
        save_dir.mkdir(parents=True)
        (save_dir / "_code_backups").mkdir(parents=True)

    ### Backup old code
    backup_types = ['.py', '.png']
    files_to_backup = [file for file in os.listdir() if any([file.endswith(t) for t in backup_types])]
    files_to_backup += [save_dir/file for file in os.listdir(save_dir) if any([file.endswith(t) for t in backup_types])]

    ### Make backup dir
    backup_dir = save_dir / "_code_backups" / ('_code_backup_'+datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
    backup_dir.mkdir(exist_ok=True)
    [shutil.copyfile(file, backup_dir/Path(str(file).split('\\')[-1])) for file in files_to_backup]


    ### Update logger code
    new_logger = MetricLogger(save_dir)
    new_logger.__dict__.update(agent.logger.__dict__)
    new_logger.reset_time()
    new_logger.init_episode()
    agent.logger = new_logger

    agent.logger.record(
                episode=agent.episode-1,
                epsilon=agent.exploration_rate*agent.exploration_modifier,
                learningrate=agent.optimizer_learningrate*agent.learningrate_modifier,
                memory=(len(agent.archive)),
                level=(0, 0),
                cur_level=0,
                step=agent.curr_step,
                save_dir=agent.save_dir,
    )



    # agent.reset() #empty memory

    # agent.logger.reset()

    # agent.exploration_rate = 0.05 # 0.50

    random.seed(time.time())



    

    ### Start actors
    # (Each performs a different function)
    print("Starting Actors:")

    print("learn")
    learn = bot_learn(agent)
    Thread(target=learn.run).start()
    print("cycle")
    cycle = bot_cycle(agent)
    Thread(target=cycle.run).start()
    print("recall")
    recall = bot_recall(agent)
    Thread(target=recall.run).start()
    print("save")
    save = bot_save(agent)
    Thread(target=save.run).start()

    ### actor_primary, actor_random, actor_render, bot_cycle, bot_learn, bot_recall, bot_save
    
    for _ in range(num_actors-1-num_random_actors):
        print("secondary")
        secondary = actor_secondary(agent)
        Thread(target=secondary.run).start()
    
    for _ in range(num_random_actors):
        print("random")
        secondary = actor_random(agent)
        Thread(target=secondary.run).start()

    print("primary")
    primary = actor_primary(agent)
    Thread(target=primary.run).start()
    
