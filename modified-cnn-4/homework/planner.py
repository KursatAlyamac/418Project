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
        
        layers = []
        
        # Initial convolution block
        layers.append(torch.nn.Conv2d(3, 128, 5, stride=2, padding=2))
        layers.append(torch.nn.BatchNorm2d(128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout2d(0.3))
        layers.append(torch.nn.MaxPool2d(2, stride=2))
        
        # Residual block (keeping same channel dimensions)
        self.conv1 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        
        # Reduction in channels after residual block
        layers.append(torch.nn.Conv2d(128, 64, 1, stride=1))
        
        # Final layers
        layers.append(torch.nn.Conv2d(64, 32, 3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout2d(0.2))
        
        layers.append(torch.nn.Conv2d(32, 1, 1, stride=1))
        
        self._conv = torch.nn.Sequential(*layers)

    def forward(self, img):
        # Initial convolutions
        x = self._conv[0:5](img)
        
        # Residual block
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + identity)  # Skip connection
        
        # Final layers
        x = self._conv[5:](x)
        
        return spatial_argmax(x[:, 0])


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
