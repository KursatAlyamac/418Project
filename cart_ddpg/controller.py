import pystk

def control(aim_point, current_vel, steer_gain=8, accel_gain = 8, skid_thresh=0.15, target_vel=50):
    #this seems to initialize an object
    action = pystk.Action()

    #compute acceleration
    if current_vel < target_vel:
        action.acceleration = accel_gain * abs((target_vel - current_vel)/target_vel)
        action.brake = False
    elif current_vel > target_vel:
        action.brake = True
        action.acceleration = 0
    else:
        action.acceleration = 0

    action.steer = steer_gain * aim_point[0]
    if abs(aim_point[0]) > skid_thresh:
        action.drift = True

    return action
    