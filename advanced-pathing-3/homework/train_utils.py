import torch
import numpy as np

class LRScheduler:
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.best_loss = None
        self.bad_epochs = 0
        
    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss:
            self.best_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            
        if self.bad_epochs >= self.patience:
            self.bad_epochs = 0
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                print(f'Reducing learning rate from {old_lr} to {new_lr}')

class ActionSmoother:
    """Smooths actions to prevent jerky steering"""
    def __init__(self, smoothing_factor=0.3):
        self.prev_action = None
        self.smoothing_factor = smoothing_factor
        
    def smooth(self, action):
        if self.prev_action is None:
            self.prev_action = action
            return action
            
        # Smooth steering
        action.steer = (1 - self.smoothing_factor) * action.steer + \
                      self.smoothing_factor * self.prev_action.steer
                      
        self.prev_action = action
        return action