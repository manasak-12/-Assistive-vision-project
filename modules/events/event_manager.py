import geopy.distance
import math

class EventManager:
    def compute_bearing(self, p1, p2):
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dLon = lon2 - lon1

        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)

        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def detect_turn(self, p_prev, p_current, p_next):
        b1 = self.compute_bearing(p_prev, p_current)
        b2 = self.compute_bearing(p_current, p_next)
        diff = (b2 - b1 + 360) % 360

        if diff < 30 or diff > 330:
            return "STRAIGHT"
        elif 30 <= diff <= 150:
            return "TURN RIGHT"
        elif 210 <= diff <= 330:
            return "TURN LEFT"
        else:
            return "SLIGHT TURN"

    def get_distance(self, p1, p2):
        return geopy.distance.distance(p1, p2).meters

    def generate_event(self, route, index):
        if index >= len(route) - 2:
            return "DESTINATION REACHED"

        p_prev = route[index]
        p_current = route[index + 1]
        p_next = route[index + 2]

        turn_type = self.detect_turn(p_prev, p_current, p_next)
        distance = int(self.get_distance(p_current, p_next))

        return f"{turn_type} IN {distance} METERS"
