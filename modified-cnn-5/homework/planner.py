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
        
        # 1. Initial convolution block
        layers.append(torch.nn.Conv2d(3, 64, 7, stride=2, padding=3))
        layers.append(torch.nn.BatchNorm2d(64))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.MaxPool2d(3, stride=2, padding=1))
        
        # 2. Residual blocks with proper dimension matching
        self.res_block1 = self._make_res_block(64, 128)
        self.downsample1 = self._make_downsample(64, 128)
        
        self.res_block2 = self._make_res_block(128, 256)
        self.downsample2 = self._make_downsample(128, 256)
        
        # 3. Final layers
        layers.append(torch.nn.AdaptiveAvgPool2d((8, 8)))
        layers.append(torch.nn.Conv2d(256, 64, 1))
        layers.append(torch.nn.BatchNorm2d(64))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Dropout2d(0.3))
        layers.append(torch.nn.Conv2d(64, 1, 1))
        
        self._conv = torch.nn.Sequential(*layers)

    def _make_res_block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )
    
    def _make_downsample(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels)
        )

    def forward(self, img):
        # Initial conv block
        x = self._conv[0:4](img)
        
        # First residual block
        identity = self.downsample1(x)
        x = self.res_block1(x)
        x = F.relu(x + identity)
        
        # Second residual block
        identity = self.downsample2(x)
        x = self.res_block2(x)
        x = F.relu(x + identity)
        
        # Final layers
        x = self._conv[4:](x)
        
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
