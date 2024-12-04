import pystk


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=30):
    import numpy as np
    action = pystk.Action()

    # Enhanced steering with adaptive gain and smoothing
    turn_sharpness = abs(aim_point[0])
    speed_factor = current_vel / target_vel
    dynamic_gain = steer_gain * (1.0 + 0.5 * speed_factor) * (1.0 + turn_sharpness)
    steering = np.clip(dynamic_gain * aim_point[0], -1, 1)
    action.steer = steering
    
    # Improved dynamic velocity targeting based on turn characteristics
    turn_penalty = turn_sharpness ** 1.5  # More aggressive turn slowdown
    local_target = target_vel * (1 - 0.6 * turn_penalty)
    vel_error = local_target - current_vel
    
    # Enhanced acceleration and brake control
    if vel_error > 0:
        action.acceleration = np.clip(vel_error / 12, 0, 1)
        action.brake = False
        # More strategic nitro usage
        action.nitro = (vel_error > 5 and turn_sharpness < 0.15 and current_vel < target_vel * 1.2)
    else:
        action.acceleration = 0
        # Progressive braking based on speed excess
        brake_intensity = min(-vel_error / 10, 1.0)
        action.brake = brake_intensity > 0.2
        action.nitro = False
    
    # Advanced drift control with speed and turn consideration
    should_drift = ((turn_sharpness > skid_thresh * 0.8 and speed_factor > 0.7) or 
                   (turn_sharpness > skid_thresh * 1.2 and speed_factor > 0.5))
    
    if should_drift:
        # Adjust target velocity during drift
        local_target *= 0.85
        # Maintain some acceleration during drift
        action.acceleration = max(action.acceleration, 0.3)
        
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
