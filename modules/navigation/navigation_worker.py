# import osmnx as ox
# import networkx as nx
# import geopy.distance
# import time

# class NavigationWorker:
#     def __init__(self, start_point, end_point):
#         """
#         start_point = (lat, lon)
#         end_point   = (lat, lon)
#         """
#         self.start_point = start_point
#         self.end_point = end_point
#         self.G = None
#         self.route_nodes = []
#         self.route_coords = []
#         self.current_index = 0

#     # -----------------------------------------
#     # Load map and compute route
#     # -----------------------------------------
#     def load_route(self):
#         print("[INFO] Downloading map and generating route...")

#         # Download walkable graph around start point
#         self.G = ox.graph_from_point(self.start_point, dist=2000, network_type='walk')

#         # Get nearest nodes in the graph
#         start_node = ox.distance.nearest_nodes(self.G, X=self.start_point[1], Y=self.start_point[0])
#         end_node   = ox.distance.nearest_nodes(self.G, X=self.end_point[1], Y=self.end_point[0])

#         # Compute shortest path
#         self.route_nodes = nx.shortest_path(self.G, start_node, end_node, weight='length')

#         # Convert nodes to coordinates
#         self.route_coords = [
#             (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
#             for node in self.route_nodes
#         ]

#         print("[INFO] Route loaded successfully.")
#         print(f"[INFO] Total points: {len(self.route_coords)}")

#     # -----------------------------------------
#     # GPS Simulation
#     # -----------------------------------------
#     def simulate_gps(self):
#         """Simulates movement along the route."""
#         if not self.route_coords:
#             print("[ERROR] Route not loaded!")
#             return None

#         # Return next point in route
#         if self.current_index < len(self.route_coords):
#             point = self.route_coords[self.current_index]
#             self.current_index += 1
#             time.sleep(1)  # simulate movement delay
#             return point
#         else:
#             return None

#     # -----------------------------------------
#     # Distance to next waypoint
#     # -----------------------------------------
#     def get_distance(self, p1, p2):
#         return geopy.distance.distance(p1, p2).meters

import geopy.distance

class NavigationWorker:
    def __init__(self, start_point, end_point):
        self.start = start_point
        self.end = end_point
        self.route_coords = []
        self.index = 0

    # LOAD ROUTE
    def load_route(self):
        print("[INFO] Downloading map and generating route...")

        lat_step = (self.end[0] - self.start[0]) / 15
        lon_step = (self.end[1] - self.start[1]) / 15

        self.route_coords = [
            (self.start[0] + i * lat_step, self.start[1] + i * lon_step)
            for i in range(15)
        ]

        self.route_coords.append(self.end)

        print("[INFO] Route loaded successfully.")
        print("[INFO] Total points:", len(self.route_coords))

    # SIMULATED GPS
    def simulate_gps(self):
        if self.index >= len(self.route_coords):
            return None
        point = self.route_coords[self.index]
        self.index += 1
        return point

    # REAL GPS UPDATE (for Vision team later)
    def update_gps(self, lat, lon):
        self.route_coords.insert(self.index, (lat, lon))
