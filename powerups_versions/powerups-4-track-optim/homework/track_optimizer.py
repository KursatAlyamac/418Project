import torch
import numpy as np
import pystk

class TrackOptimizer:
    def __init__(self, track_length=1000):
        """
        Initialize track optimizer to help with path planning
        track_length: number of points to sample along track
        """
        self.track_length = track_length
        self.track_points = None
        self.track_width = None
        self.track_distance = None
        self.initialized = False

    def initialize_track(self, track):
        """
        Initialize track data from pystk Track object
        """
        # Sample track points and width
        self.track_points = track.path_nodes.reshape(-1, 2, 3)  # N x 2 x 3 -> path segments
        self.track_width = track.path_width
        self.track_distance = track.path_distance.reshape(-1, 2)
        
        # Interpolate track points for smoother path
        t = np.linspace(0, 1, self.track_length)
        points = []
        widths = []
        
        for i in range(len(self.track_points)-1):
            p1, p2 = self.track_points[i], self.track_points[i+1]
            w1, w2 = self.track_width[i], self.track_width[i+1]
            
            # Linear interpolation
            interp_points = np.array([p1[0] + t*(p2[0] - p1[0]) for t in t])
            interp_width = w1 + t*(w2 - w1)
            
            points.append(interp_points)
            widths.append(interp_width)
            
        self.track_points = np.concatenate(points)
        self.track_width = np.concatenate(widths)
        self.initialized = True

    def get_optimal_trajectory(self, kart_location, look_ahead=10):
        """
        Get optimal racing line points ahead of kart
        """
        if not self.initialized:
            return None
            
        # Find closest track point to kart
        kart_pos = np.array(kart_location)
        distances = np.linalg.norm(self.track_points - kart_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get look_ahead points
        trajectory_points = []
        for i in range(look_ahead):
            idx = (closest_idx + i) % len(self.track_points)
            point = self.track_points[idx]
            width = self.track_width[idx]
            trajectory_points.append((point, width))
            
        return trajectory_points

    def get_target_point(self, kart_location, kart_rotation):
        """
        Get target point for steering based on optimal trajectory
        """
        if not self.initialized:
            return None
            
        # Get trajectory points
        trajectory = self.get_optimal_trajectory(kart_location)
        if not trajectory:
            return None
            
        # Convert kart rotation quaternion to forward vector
        forward = quaternion_to_forward(kart_rotation)
        
        # Find best target point considering track curvature
        best_point = None
        best_score = float('-inf')
        
        for point, width in trajectory:
            # Vector from kart to point
            to_point = point - np.array(kart_location)
            distance = np.linalg.norm(to_point)
            
            if distance < 1e-6:
                continue
                
            # Normalize
            to_point /= distance
            
            # Score based on alignment with forward direction and distance
            alignment = np.dot(forward, to_point)
            score = alignment / (1 + 0.1 * distance)
            
            if score > best_score:
                best_score = score
                best_point = point
                
        return best_point

def quaternion_to_forward(q):
    """Convert quaternion to forward vector"""
    w, x, y, z = q
    return np.array([
        2 * (x*z + w*y),
        2 * (y*z - w*x),
        1 - 2 * (x*x + y*y)
    ]) 