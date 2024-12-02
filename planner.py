import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pystk

class AimPointPredictor(nn.Module):
    def __init__(self):
        super(AimPointPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv_output_size = self._get_conv_output((3, 224, 224))
        self.fc1 = nn.Linear(self.conv_output_size, 500)
        self.fc2 = nn.Linear(500, 2)  # Output is the aim point (x, y)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = AimPointPredictor()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_aim_point(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    aim_point = output.numpy().flatten()
    return aim_point

def control(aim_point, current_velocity):
    action = pystk.Action()
    action.steer = np.clip(aim_point[0], -1, 1)
    target_speed = 20.0 
    if current_velocity < target_speed:
        action.acceleration = 1.0
        action.brake = False
    else:
        action.acceleration = 0.0
        action.brake = True
    action.drift = abs(aim_point[0]) > 0.5
    action.nitro = False
    return action

def main():
    # Initialize PySuperTuxKart
    config = pystk.GraphicsConfig.hd()
    config.screen_width = 640
    config.screen_height = 480
    config.render_window = False  #game window
    pystk.init(config)

    track = 'zengarden'  

    # Set up the race
    race_config = pystk.RaceConfig(num_kart=1, track=track)
    race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
    race = pystk.Race(race_config)
    race.start()

    try:
        for t in range(1000):
            s = race.step()
            image = np.array(race.render_data[0].image)
            aim_point = predict_aim_point(image)
            current_velocity = np.linalg.norm(race.karts[0].velocity)
            action = control(aim_point, current_velocity)
            race.step(action)
    finally:
        race.stop()
        pystk.clean()

if __name__ == '__main__':
    main()