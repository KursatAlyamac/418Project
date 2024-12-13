import pystk
import numpy as np


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=30, turn_sensitivity=0.5):
    """
    Improved controller for SuperTuxKart.
    - Adjusts speed and steering dynamically based on aim point and current velocity.
    - Enables drifting on sharp turns and uses nitro for boosts on straight paths.
    """
    action = pystk.Action()

    # Adjust steering dynamically based on the aim point
    action.steer = np.clip(steer_gain * aim_point[0], -1, 1)

    # Engage drift for sharp turns
    action.drift = abs(aim_point[0]) > skid_thresh

    # Reduce target velocity for sharper turns
    dynamic_target_vel = target_vel * (1 - turn_sensitivity * abs(aim_point[0]))

    # Adjust acceleration and braking based on dynamic target velocity
    if current_vel < dynamic_target_vel:
        action.acceleration = 1
        action.brake = False
    else:
        action.acceleration = 0
        action.brake = current_vel > dynamic_target_vel + 5

    # Use nitro on straight paths for speed boost
    action.nitro = 0.9 * dynamic_target_vel <= current_vel < dynamic_target_vel + 5

    # Debugging: Print control parameters for monitoring and tuning
    print(f"Aim: {aim_point}, Steer: {action.steer:.2f}, Vel: {current_vel:.2f}, Target Vel: {dynamic_target_vel:.2f}")

    return action


if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            print(f"Testing on track: {t}")
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(f"Track: {t}, Steps: {steps}, Distance: {how_far}")
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
