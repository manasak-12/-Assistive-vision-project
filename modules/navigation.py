# navigation.py
# Simple navigation advice based on person bbox center and distance.
# Returns directions: "stop", "move left", "move right", "move forward", "clear"

def navigation_advice(bbox, frame_width, distance_m, danger_distance=1.2, forward_thresh=2.5):
    """
    bbox: [x1,y1,x2,y2]
    frame_width: width of frame
    distance_m: float meters or None
    returns string advice
    """
    if bbox is None:
        return "clear"

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    center_norm = (cx - frame_width/2) / (frame_width/2)  # -1..1

    if distance_m is not None and distance_m < danger_distance:
        return "stop"

    if distance_m is not None and distance_m < forward_thresh:
        # person is ahead but not immediate: guide left/right to avoid
        if center_norm < -0.2:
            return "move left"
        elif center_norm > 0.2:
            return "move right"
        else:
            return "move forward"

    # default: clear / follow
    return "clear"
