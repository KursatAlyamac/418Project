import pystk
import numpy as np

def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    """
    Enhanced controller with advanced path planning integration
    """
    action = pystk.Action()

    # Calculate steering with dynamic gain based on speed
    speed_factor = min(1.0, current_vel / target_vel)
    aim_magnitude = np.linalg.norm(aim_point)
    dynamic_steer_gain = steer_gain * (1.0 + 0.7 * (1.0 - speed_factor)) * (1.0 + aim_magnitude)
    steering = np.clip(dynamic_steer_gain * aim_point[0], -1, 1)
    action.steer = steering
    
    # Enhanced velocity control
    vel_error = target_vel - current_vel
    
    # Progressive acceleration/braking
    if vel_error > 0:
        # Smoother acceleration curve
        action.acceleration = np.clip(np.sqrt(vel_error) / 4, 0, 1)
        action.brake = False
        # Smart nitro usage
        action.nitro = vel_error > 10 and abs(steering) < 0.5
    else:
        # Progressive braking
        action.acceleration = 0
        action.brake = vel_error < -5
        action.nitro = False
    
    # Enhanced drift control
    turn_sharpness = abs(aim_point[0])
    speed_factor = current_vel / target_vel
    
    # Dynamic drift threshold based on speed
    dynamic_drift_thresh = skid_thresh * (1.0 - 0.3 * speed_factor)
    should_drift = (turn_sharpness > dynamic_drift_thresh) and (speed_factor > 0.8)
    
    # Add drift recovery
    if should_drift and abs(steering) > 0.8:
        action.acceleration *= 0.5  # Reduce acceleration while drifting
    
    action.drift = should_drift

    return action

    




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
