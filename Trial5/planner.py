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
        layers = [
            # First convolutional block
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # Input: (B, 3, 96, 128)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),                # Output: (B, 32, 24, 32)

            # Second convolutional block
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),                # Output: (B, 64, 12, 16)

            # Third convolutional block
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # Fourth convolutional block
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),               # Output: (B, 256, 6, 8)

            # Fifth convolutional block (optional, for deeper learning)
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # Fully connected layers for regression
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 6 * 8, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)  # Output: aim point coordinates (B, 2)
        ]

        self._conv = torch.nn.Sequential(*layers)

    def forward(self, img):
        x = self._conv(img)
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