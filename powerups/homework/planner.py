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
        
        # Encoder
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        
        # Path planning head
        self.path_conv = torch.nn.Conv2d(128, 1, kernel_size=1)
        
        # Powerup detection head
        self.powerup_conv = torch.nn.Conv2d(128, 16, kernel_size=3, padding=1)
        self.powerup_classifier = torch.nn.Conv2d(16, 1, kernel_size=1)
        
        # Activation functions
        self.relu = torch.nn.ReLU()

    def forward(self, img, return_powerup=False):
        """
        :param img: Input image
        :param return_powerup: If True, return both aim_point and powerup detection. If False, only return aim_point
        :return: aim_point tensor or (aim_point, powerup_present) tuple depending on return_powerup
        """
        # Shared encoder
        x = self.relu(self.conv1(img))
        x = self.relu(self.conv2(x))
        features = self.relu(self.conv3(x))
        
        # Path planning
        path = self.path_conv(features)
        aim_point = spatial_argmax(path[:, 0])
        
        if not return_powerup:
            return aim_point
            
        # Powerup detection
        powerup = self.powerup_conv(features)
        powerup = self.powerup_classifier(powerup)
        powerup_present = torch.sigmoid(powerup.mean(dim=[2,3]))
        
        return aim_point, powerup_present


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
