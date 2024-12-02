import numpy as np
import pystk

def control(aim_point, current_velocity):
    """
    Compute the action for the kart based on the aim point and current velocity.

    Parameters:
    - aim_point: A tuple or list with two elements (x, y), where x and y are in the range [-1, 1].
    - current_velocity: A float representing the current speed of the kart.

    Returns:
    - action: An instance of pystk.Action with the computed control inputs.
    """
    action = pystk.Action()

    action.steer = np.clip(aim_point[0] * 2.0, -1.0, 1.0) 

    target_speed = 20.0  
    speed_error = target_speed - current_velocity

    if speed_error > 0:
        action.acceleration = np.clip(speed_error / target_speed, 0.0, 1.0)
        action.brake = False
    else:
        action.acceleration = 0.0
        action.brake = True

    if abs(aim_point[0]) > 0.5:
        action.drift = True
    else:
        action.drift = False

    action.nitro = False  # Set to True if you want to use nitro under certain conditions

    return action

def main():
    # Initialize PySuperTuxKart
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 640
    config.screen_height = 480
    config.render_window = True  #game window
    pystk.init(config)

    track = 'zengarden'  #  track name

    # Set up the race
    race_config = pystk.RaceConfig(num_kart=1, track=track)
    race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    race = pystk.Race(race_config)
    race.start()

    try:
        for t in range(1000):
            s = race.step()
            aim_point = get_aim_point_from_game_state(race)
            current_velocity = np.linalg.norm(race.karts[0].velocity)
            action = control(aim_point, current_velocity)

            race.step(action)
    finally:
        race.stop()
        pystk.clean()

def get_aim_point_from_game_state(race):
    """
    Placeholder function to compute aim point from the game state.

    Replace this function with your actual logic to compute the aim point,
    possibly using a planner or other method.

    Returns:
    - aim_point: A tuple (x, y) with values in [-1, 1]
    """
    # Example: aim straight ahead
    return (0.0, 0.0)

if __name__ == '__main__':
    main()