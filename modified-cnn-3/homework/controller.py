import pystk


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    import numpy as np
    #this seems to initialize an object
    action = pystk.Action()

    steering = np.clip(steer_gain * aim_point[0], -1, 1)
    action.steer = steering
    
    vel_error = target_vel - current_vel
    
    if vel_error > 0:
        action.acceleration = np.clip(vel_error / 20, 0, 1)
        action.brake = False
        action.nitro = vel_error > 10
    else:
        action.acceleration = 0
        action.brake = vel_error < -5
        action.nitro = False
    
    turn_sharpness = abs(aim_point[0])
    speed_factor = current_vel / target_vel
    should_drift = (turn_sharpness > skid_thresh) and (speed_factor > 0.8)
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
