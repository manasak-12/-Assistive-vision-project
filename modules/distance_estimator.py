# distance_estimator.py
# Estimate distance (meters) from a depth map and bounding box.
# The depth_map is expected to be normalized 0-255 (as our MiDaS worker returns),
# and you will need to calibrate scale_to_meters factor for real measurements.

import numpy as np
import cv2

class DistanceEstimator:
    def __init__(self, scale_to_meters=0.02):
        """
        scale_to_meters: multiply normalized depth (0-255) by scale to get meters.
        Default 0.02 => 255*0.02=5.1m max depth (adjust as needed).
        """
        self.scale = scale_to_meters

    def bbox_distance(self, depth_map, bbox, use_median=True, pad=5):
        """
        depth_map: uint8 depth map (0..255)
        bbox: [x1,y1,x2,y2]
        returns distance in meters (float) or None
        """
        if depth_map is None:
            return None

        h, w = depth_map.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1)-pad)
        y1 = max(0, int(y1)-pad)
        x2 = min(w-1, int(x2)+pad)
        y2 = min(h-1, int(y2)+pad)

        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # convert to float
        vals = crop.astype(np.float32).flatten()

        # discard zeros and extreme values
        vals = vals[vals > 0]
        if vals.size == 0:
            return None

        if use_median:
            depth_val = np.median(vals)
        else:
            depth_val = np.mean(vals)

        meters = float(depth_val) * self.scale
        return meters
