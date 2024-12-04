import pystk
import numpy as np
from train_utils import ActionSmoother

class EnhancedController:
    def __init__(self):
        self.action_smoother = ActionSmoother(smoothing_factor=0.2)
        self.prev_steering = 0
        self.recovery_timer = 0
        self.drift_cooldown = 0
        
    def __call__(self, aim_point, current_vel, steer_gain=8, skid_thresh=0.15, target_vel=25):
        """
        Enhanced controller with improved turn handling
        """
        action = pystk.Action()

        # Calculate steering with dynamic gain based on speed
        speed_factor = min(1.0, current_vel / target_vel)
        aim_magnitude = np.linalg.norm(aim_point)
        
        # Increased steering response for sharper turns
        turn_sharpness = abs(aim_point[0])
        
        # More aggressive steering gain for sharp turns
        dynamic_steer_gain = steer_gain * (1.0 + 1.2 * (1.0 - speed_factor)) * \
                            (1.0 + turn_sharpness)  # Increased turn sensitivity
        
        # Calculate raw steering with enhanced response
        raw_steering = np.clip(dynamic_steer_gain * aim_point[0], -1, 1)
        
        # Reduce steering interpolation for sharper response
        steering_change = raw_steering - self.prev_steering
        max_steering_change = 0.15 * (1.0 + speed_factor)  # Allow faster steering changes
        steering = self.prev_steering + np.clip(steering_change, -max_steering_change, max_steering_change)
        self.prev_steering = steering
        
        action.steer = steering
        
        # Enhanced velocity control for turns
        vel_error = target_vel - current_vel
        
        # More aggressive speed reduction in turns
        turn_speed_factor = 1.0 - 0.7 * turn_sharpness  # Increased turn speed reduction
        adjusted_target = target_vel * turn_speed_factor
        
        # Early braking for turns
        if turn_sharpness > 0.3 and current_vel > adjusted_target:
            vel_error = adjusted_target - current_vel
        
        # Progressive acceleration/braking
        if vel_error > 0:
            # Reduced acceleration in sharp turns
            turn_accel_factor = 1.0 - 0.5 * turn_sharpness
            action.acceleration = np.clip(np.sqrt(vel_error) / 4, 0, 1) * turn_accel_factor
            action.brake = False
            # More conservative nitro usage
            action.nitro = vel_error > 10 and turn_sharpness < 0.3 and self.recovery_timer == 0
        else:
            # More aggressive braking in turns
            brake_factor = 1.0 + turn_sharpness
            action.acceleration = 0
            action.brake = vel_error * brake_factor < -3
            action.nitro = False
        
        # Enhanced drift control
        if self.drift_cooldown > 0:
            self.drift_cooldown -= 1
        
        # More aggressive drift triggering
        speed_factor = current_vel / target_vel
        dynamic_drift_thresh = skid_thresh * (1.0 - 0.4 * speed_factor)  # Lower threshold for drifting
        should_drift = (turn_sharpness > dynamic_drift_thresh) and (speed_factor > 0.7)
        
        if should_drift and self.drift_cooldown == 0:
            if abs(steering) > 0.7:  # Reduced threshold for drift acceleration reduction
                action.acceleration *= 0.6  # Less acceleration reduction while drifting
                self.recovery_timer = 8  # Shorter recovery period
                self.drift_cooldown = 15  # Add drift cooldown
            action.drift = True
        else:
            action.drift = False
            
        # Recovery behavior
        if self.recovery_timer > 0:
            action.acceleration *= 0.8  # Less aggressive acceleration reduction during recovery
            self.recovery_timer -= 1
            
        # Apply action smoothing
        return self.action_smoother.smooth(action)

class ActionSmoother:
    """Smooths actions to prevent jerky steering"""
    def __init__(self, smoothing_factor=0.2):
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

# Create a global controller instance
controller = EnhancedController()

def control(aim_point, current_vel, steer_gain=8, skid_thresh=0.15, target_vel=25):
    """Wrapper function for the enhanced controller"""
    return controller(aim_point, current_vel, steer_gain, skid_thresh, target_vel)

    




if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
