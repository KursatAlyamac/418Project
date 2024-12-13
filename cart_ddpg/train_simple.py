from env import TrackEnv
from ddpg_simple import DDPG
from controller import control
import pystk
import pandas as pd

environment = TrackEnv()
agent = DDPG()
num_episodes = 100
batch_size = 32
track = ""

initial_action = pystk.Action()
initial_action.steer = 0
initial_action.acceleration = 0
initial_action.brake = True
initial_action.drift = False

actor_losses = []
critic_losses = []
velocities = []
times = []


for episode in range(num_episodes):
    print(episode)
    info = environment.step(track, initial_action)

    image = info['image_tensor']
    action = info['action_tensor']
    aim_point_tensor = info['aim_point_tensor']

    done = False
    while not done:
        accel_action = agent.select_action(image, aim_point_tensor)
        current_vel = environment.get_current_vel()
        velocities.append(current_vel)
        aim_point = environment.get_aim_point()
        action = control(aim_point, current_vel)
        action.acceleration = accel_action
        print(action)
        print(f"t:{environment.t}")
        t = environment.t
        info = environment.step(track, action)

        next_image = info['image_tensor']
        action = info['action_tensor']
        next_aim_point = info['aim_point_tensor']
        reward = info['reward']
        done = info['done']
        agent.store_transition(image, aim_point_tensor, action, reward, next_image, next_aim_point, done)
        actor_loss, critic_loss = agent.update(batch_size)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        image = next_image
        aim_point_tensor = next_aim_point

    df_loss = pd.DataFrame(columns = ['Actor Loss', 'Critic Loss', 'Velocities'])
    df_times = pd.DataFrame(columns = ['Times'])    
    times.append(t)
    df_loss['Actor Loss'] = actor_losses
    df_loss['Critic Loss'] = critic_losses
    df_loss['Velocities'] = velocities
    df_times['Times'] = times

    save_path = '/projectnb/abagbind/ab_ag_bind/ec/cart_ddpg/ddpg_loss_4.csv'
    df_loss.to_csv(save_path, index=False)

    save_path = '/projectnb/abagbind/ab_ag_bind/ec/cart_ddpg/ddpg_times_4.csv'
    df_times.to_csv(save_path, index=False)
