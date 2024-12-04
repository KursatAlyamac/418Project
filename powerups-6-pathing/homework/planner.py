import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Enhanced Encoder with deeper architecture
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        
        # Additional convolutional layer for better feature extraction
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(512)
        
        # Multiple residual blocks for better feature processing
        self.res1 = ResidualBlock(512, 512)
        self.res2 = ResidualBlock(512, 512)
        self.res3 = ResidualBlock(512, 512)
        
        # Path planning head with enhanced attention
        self.path_attention = MultiScaleAttention(512)
        self.path_conv = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 1, kernel_size=1)
        )
        
        # Enhanced powerup detection head
        self.powerup_conv1 = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.powerup_bn1 = torch.nn.BatchNorm2d(256)
        self.powerup_conv2 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.powerup_bn2 = torch.nn.BatchNorm2d(128)
        self.powerup_classifier = torch.nn.Conv2d(128, 1, kernel_size=1)
        
        # Add racing line prediction head
        self.racing_line = RacingLinePredictor(512)
        
        self.dropout = torch.nn.Dropout(0.3)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, img, return_powerup=False):
        # Encoder path
        x = self.leaky_relu(self.bn1(self.conv1(img)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Get racing features and transform them for integration
        racing_features = self.racing_line(x)
        racing_influence = racing_features.mean(dim=[2, 3])  # [B, C]
        # Project racing influence to 2D space for aim point adjustment
        racing_projection = torch.nn.functional.linear(
            racing_influence, 
            torch.nn.Parameter(torch.randn(2, racing_influence.shape[1]).to(racing_influence.device))
        )  # [B, 2]
        
        # Get path features and aim point
        path_features = self.path_attention(x + racing_features)
        path = self.path_conv(path_features)
        aim_point = spatial_argmax(path[:, 0])
        
        # Integrate racing line with aim point (now dimensions match)
        aim_point = aim_point + 0.1 * torch.tanh(racing_projection)
        
        if not return_powerup:
            return aim_point
            
        powerup = self.powerup_conv1(x)
        powerup = self.dropout(powerup)
        powerup = self.leaky_relu(self.powerup_bn2(self.powerup_conv2(powerup)))
        powerup = self.dropout(powerup)
        powerup = self.powerup_classifier(powerup)
        powerup_present = torch.sigmoid(powerup.mean(dim=[2,3]))
        
        return aim_point, powerup_present, racing_features


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x


class MultiScaleAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.combine = torch.nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        
    def forward(self, x):
        att1 = torch.sigmoid(self.conv1(x))
        att3 = torch.sigmoid(self.conv3(x))
        att5 = torch.sigmoid(self.conv5(x))
        
        multi_scale = torch.cat([att1 * x, att3 * x, att5 * x], dim=1)
        return self.combine(multi_scale)


class RacingLinePredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(in_channels//2)
        self.conv2 = torch.nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(in_channels//2)
        self.conv3 = torch.nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)