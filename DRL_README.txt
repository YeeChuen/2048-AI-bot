All files found in DRL folder

Training setting found in agent.py and main.py

MCTS setting found in env.py

Network settings found in net.py and main.py

Checkpoint automatically saved. Checkpoint can be loaded by modifying main.py

cuda can be disabled in main.py

To test the MCTS DRL agent, set self.use_mcts to True in env.py otherwise set it to False to run non-MCTS DRL agent.

Requires:
	tqdm
	numpy
	matplotlib
	gym
	pytorch