import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


path = '/projectnb/abagbind/ab_ag_bind/ec/cart_ddpg/ddpg_times_2.csv'
df = pd.read_csv(path)


times = df['Times'].to_numpy()
plt.plot(range(len(times)), times, label='DDPG Agent')
plt.axhline(y=597, color='r', linestyle='--', label='Baseline Controller')

plt.grid(True)
plt.title('DDPG Agent Time to Complete Track')
plt.xlabel('Episode')
plt.ylabel('Frames to Complete Track')

plt.legend()
plt.show()


# path = '/projectnb/abagbind/ab_ag_bind/ec/cart_ddpg/ddpg_loss_3.csv'
# df = pd.read_csv(path)

# actor_loss = df['Actor Loss']
# critic_loss = df['Critic Loss']
# velocity = df['Velocities'].to_numpy()

# print(np.max(velocity))

# path = '/projectnb/abagbind/ab_ag_bind/ec/cart_ddpg/ddpg_times_3.csv'
# df = pd.read_csv(path)
# times = df['Times'].to_numpy()
# cumulative_times = np.cumsum(times)


# plt.plot(range(len(velocity[cumulative_times[5]:cumulative_times[8]])), velocity[cumulative_times[5]:cumulative_times[8]], label = 'DDPG Agent')
# processes = [
#     {"start": 0, "end": times[6]},
#     {"start": times[6], "end": times[7] + times[6]},
#     {"start": times[7] + times[6], "end": times[8] + times[7] + times[6]}
# ]
# print(times[6], times[7], times[8])
# # Adding vertical lines for process boundaries
# for process in processes:
#     plt.axvline(x=process['start'], color='green', linestyle='--', label='Start Track' if process == processes[0] else "")
#     # plt.axvline(x=process['end'], color='red', linestyle='-.', label='End' if process == processes[0] else "")

# plt.grid(True)
# plt.title('DDPG Agent Kart Velocity Profile')
# plt.xlabel('Frame')
# plt.ylabel('Kart Velocity')
# plt.legend()
# plt.show()

