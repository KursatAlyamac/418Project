import pystk
import numpy as np


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=30, turn_sensitivity=0.5):
    """
    Advanced controller for SuperTuxKart.
    - Dynamically adjusts steering, acceleration, and braking.
    - Intelligent use of drifting, nitro, and speed adjustments based on kart dynamics.
    """
    action = pystk.Action()

    # Adjust steering dynamically based on aim point
    action.steer = np.clip(steer_gain * aim_point[0], -1, 1)

    # Adaptive skid threshold based on velocity
    adaptive_skid_thresh = skid_thresh + 0.1 * (current_vel / target_vel)
    action.drift = abs(aim_point[0]) > adaptive_skid_thresh

    # Adaptive target velocity based on turn angle
    dynamic_target_vel = target_vel * (1 - turn_sensitivity * abs(aim_point[0]))

    # Acceleration and braking logic
    if current_vel < dynamic_target_vel:
        action.acceleration = 1
        action.brake = False
    elif current_vel > dynamic_target_vel + 5:
        action.acceleration = 0
        action.brake = True
    else:
        action.acceleration = 0.5  # Smooth acceleration for minor adjustments
        action.brake = False

    # Nitro usage: Use nitro only if kart is stable and on a straight path
    straight_path = abs(aim_point[0]) < 0.1
    action.nitro = straight_path and dynamic_target_vel * 0.9 <= current_vel <= dynamic_target_vel + 5

    # Debugging information for fine-tuning
    print(f"Aim: {aim_point}, Steer: {action.steer:.2f}, Vel: {current_vel:.2f}, "
          f"Target Vel: {dynamic_target_vel:.2f}, Nitro: {action.nitro}")

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
