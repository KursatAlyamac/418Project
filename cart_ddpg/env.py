import torch
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

import dense_transforms
#Import pytsk
import pystk


#Environment Class
import matplotlib.pyplot as plt



class TrackEnv():

    def __init__(self, screen_width=128, screen_height=96):
        TrackEnv._singleton = self
        fig, ax = plt.subplots(1, 1)
        self.ax = ax
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.race=None
        self.race_buffer = []

    def _point_on_track(self,distance,track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """

        node_idx = np.searchsorted(track.path_distance[..., 1],
                                   distance % track.path_distance[-1, 1]) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    def _to_image(self,x, proj, view):
        # input x is return of _point_on_track, return of to_image is aim point
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)
    
    def start_race(self,track):
        config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        self.race = pystk.Race(config)
        self.race.start()
        self.last_rescue_t = 0 # initalize last rescue
        self.t = 0 # time of starting the race
        self.state = pystk.WorldState()
        self.track = pystk.Track()
        self.last_kart_distance = 0
        self.reset_track = False

    
    def get_current_vel(self):
        #find current velocity
        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)

        return current_vel
    
    def get_aim_point(self):
         #calculate variables needed for aim pointer
        proj = np.array(self.state.players[0].camera.projection).T
        view = np.array(self.state.players[0].camera.view).T
    
        #calculate aim pointer
        kart = self.state.players[0].kart

        aim_point_world = self._point_on_track(kart.distance_down_track+15, self.track)
        aim_point_image = np.round(self._to_image(aim_point_world, proj, view),1)

        return aim_point_image
    
    def step(self,track,action):
        if self.race is not None and self.race.config.track!= track: 
            self.start_race(track)
  
        #If no race has been done yet then start a race
        if self.race is None:
            print("starting race again")
            self.start_race(track)
            
        # initalize reward for each step as 0
        reward = 0 

        self.race.step(action)

        self.state.update()
        self.track.update()

        #find current velocity
        kart = self.state.players[0].kart
        current_vel = np.linalg.norm(kart.velocity)

        #calculate variables needed for aim pointer
        proj = np.array(self.state.players[0].camera.projection).T
        view = np.array(self.state.players[0].camera.view).T
    
        #calculate aim pointer
        aim_point_world = self._point_on_track(kart.distance_down_track+15, self.track)
        aim_point_image = np.round(self._to_image(aim_point_world, proj, view),1)

        self.t = self.t+1

        if (current_vel < 1) and self.t - self.last_rescue_t > 30:
            self.last_rescue_t = self.t

            initial_action = pystk.Action()
            initial_action.rescue = True
            initial_action.steer = 0
            initial_action.acceleration = 0
            initial_action.brake = True
            initial_action.drift = False
            self.race.step(initial_action)

            reward -= 50 
        
        #See if done
        done = self.state.players[0].kart.race_result
        if done == True:
            reward+=50

        reward -= (abs(aim_point_image[0]) / 10)

        reward += kart.distance_down_track - self.last_kart_distance

        reward += action.acceleration

        self.last_kart_distance = kart.distance_down_track

        image = np.array(self.race.render_data[0].image)
        image_tensor = dense_transforms.ToTensor()(image)[0]
        
        action_tensor = torch.tensor([action.steer])

        aim_point_tensor = torch.tensor(aim_point_image, dtype = torch.float32)
        reward_tensor = torch.tensor(reward, dtype = torch.float32)
        done_tensor = torch.tensor(done, dtype = torch.float32)

        print(f"reward: {reward}")
        
        self.ax.clear()
        self.ax.imshow(self.race.render_data[0].image)
        WH2 = np.array([self.config.screen_width, self.config.screen_height]) / 2
        self.ax.add_artist(plt.Circle(WH2*(1+self._to_image(kart.location, proj, view)), 2, ec='b', fill=False, lw=1.5))
        self.ax.add_artist(plt.Circle(WH2*(1+self._to_image(aim_point_world, proj, view)), 2, ec='r', fill=False, lw=1.5))
        plt.pause(1e-3)

        if done == True:
            self.reset(track)

        return {'image_tensor':image_tensor,
               'action_tensor':action_tensor,
               'aim_point_tensor': aim_point_tensor,
               'reward':reward_tensor,
               'done': done_tensor}

    def reset(self,track):
        self.race.stop()
        del self.race
        self.race = None
        self.start_race(track)

