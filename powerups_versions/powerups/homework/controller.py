import pystk


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=30):
    import numpy as np
    action = pystk.Action()

    # Improved steering with dynamic gain based on speed
    speed_factor = current_vel / target_vel
    dynamic_gain = steer_gain * (1.0 + 0.5 * speed_factor)
    steering = np.clip(dynamic_gain * aim_point[0], -1, 1)
    action.steer = steering
    
    # Dynamic target velocity based on turn sharpness
    turn_sharpness = abs(aim_point[0])
    local_target = target_vel * (1 - 0.5 * turn_sharpness)
    vel_error = local_target - current_vel
    
    # Acceleration and brake control
    if vel_error > 0:
        action.acceleration = np.clip(vel_error / 15, 0, 1)
        action.brake = False
        # Use nitro on straightaways when we're below target speed
        action.nitro = vel_error > 5 and turn_sharpness < 0.2
    else:
        action.acceleration = 0
        action.brake = vel_error < -3
        action.nitro = False
    
    # Improved drift control
    should_drift = (turn_sharpness > skid_thresh) and (speed_factor > 0.7)
    if should_drift:
        # Reduce target velocity during drift
        local_target *= 0.8
        
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
