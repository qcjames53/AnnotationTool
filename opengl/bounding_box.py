import math

# Bounding Box Object Class
class BoundingBox():
    """
    A class to represent a 3D box placed along the ground plane at a height of 0
    """
    def __init__(self, type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, color_value):
        self.type = type
        self.truncated = truncated
        self.occluded = occluded
        self.alpha = alpha
        self.bbox = bbox
        self.dimensions = [dimensions[0], dimensions[1], dimensions[2]]
        self.location = [location[0], location[1], location[2]]
        self.rotation_y = rotation_y

        self.color_value = color_value
        self.vertices = None
        self.build_vertices()

    def build_vertices(self):
        v0_x = self.location[0] + self.dimensions[1] / 2
        v0_y = self.location[1]
        v0_z = self.location[2] - self.dimensions[2] / 2
        v0_x2 = (self.location[0] + (((v0_x-self.location[0]) * math.cos(self.rotation_y)) - ((v0_z-self.location[2]) * math.sin(self.rotation_y))))
        v0_z2 = (self.location[2] + (((v0_z-self.location[2]) * math.cos(self.rotation_y)) + ((v0_x-self.location[0]) * math.sin(self.rotation_y))))
        v0 = (v0_x2,v0_y,v0_z2)

        v1_x = self.location[0] + self.dimensions[1] / 2
        v1_y = self.location[1] - self.dimensions[0]
        v1_z = self.location[2] - self.dimensions[2] / 2
        v1_x2 = self.location[0] + ((v1_x-self.location[0]) * math.cos(self.rotation_y) - (v1_z-self.location[2]) * math.sin(self.rotation_y))
        v1_z2 = self.location[2] + ((v1_z-self.location[2]) * math.cos(self.rotation_y) + (v1_x-self.location[0]) * math.sin(self.rotation_y))
        v1 = (v1_x2, v1_y, v1_z2)

        v2_x = self.location[0] - self.dimensions[1] / 2
        v2_y = self.location[1] - self.dimensions[0]
        v2_z = self.location[2] - self.dimensions[2] / 2
        v2_x2 = self.location[0] + ((v2_x-self.location[0]) * math.cos(self.rotation_y) - (v2_z-self.location[2]) * math.sin(self.rotation_y))
        v2_z2 = self.location[2] + ((v2_z-self.location[2]) * math.cos(self.rotation_y) + (v2_x-self.location[0]) * math.sin(self.rotation_y))
        v2 = (v2_x2, v2_y, v2_z2)

        v3_x = self.location[0] - self.dimensions[1] / 2
        v3_y = self.location[1]
        v3_z = self.location[2] - self.dimensions[2] / 2
        v3_x2 = self.location[0] + ((v3_x-self.location[0]) * math.cos(self.rotation_y) - (v3_z-self.location[2]) * math.sin(self.rotation_y))
        v3_z2 = self.location[2] + ((v3_z-self.location[2]) * math.cos(self.rotation_y) + (v3_x-self.location[0]) * math.sin(self.rotation_y))
        v3 = (v3_x2, v3_y, v3_z2)

        v4_x = self.location[0] + self.dimensions[1] / 2
        v4_y = self.location[1]
        v4_z = self.location[2] + self.dimensions[2] / 2
        v4_x2 = self.location[0] + ((v4_x-self.location[0]) * math.cos(self.rotation_y) - (v4_z-self.location[2]) * math.sin(self.rotation_y))
        v4_z2 = self.location[2] + ((v4_z-self.location[2]) * math.cos(self.rotation_y) + (v4_x-self.location[0]) * math.sin(self.rotation_y))
        v4 = (v4_x2, v4_y, v4_z2)

        v5_x = self.location[0] + self.dimensions[1] / 2
        v5_y = self.location[1] - self.dimensions[0]
        v5_z = self.location[2] + self.dimensions[2] / 2
        v5_x2 = self.location[0] + ((v5_x-self.location[0]) * math.cos(self.rotation_y) - (v5_z-self.location[2]) * math.sin(self.rotation_y))
        v5_z2 = self.location[2] + ((v5_z-self.location[2]) * math.cos(self.rotation_y) + (v5_x-self.location[0]) * math.sin(self.rotation_y))
        v5 = (v5_x2, v5_y, v5_z2)

        v6_x = self.location[0] - self.dimensions[1] / 2
        v6_y = self.location[1]
        v6_z = self.location[2] + self.dimensions[2] / 2
        v6_x2 = self.location[0] + ((v6_x-self.location[0]) * math.cos(self.rotation_y) - (v6_z-self.location[2]) * math.sin(self.rotation_y))
        v6_z2 = self.location[2] + ((v6_z-self.location[2]) * math.cos(self.rotation_y) + (v6_x-self.location[0]) * math.sin(self.rotation_y))
        v6 = (v6_x2, v6_y, v6_z2)

        v7_x = self.location[0] - self.dimensions[1] / 2
        v7_y = self.location[1] - self.dimensions[0]
        v7_z = self.location[2] + self.dimensions[2] / 2
        v7_x2 = self.location[0] + ((v7_x-self.location[0]) * math.cos(self.rotation_y) - (v7_z-self.location[2]) * math.sin(self.rotation_y))
        v7_z2 = self.location[2] + ((v7_z-self.location[2]) * math.cos(self.rotation_y) + (v7_x-self.location[0]) * math.sin(self.rotation_y))
        v7 = (v7_x2, v7_y, v7_z2)

        v8_x = self.location[0]
        v8_y = self.location[1] - self.dimensions[0] / 2
        v8_z = self.location[2] + self.dimensions[2] / 2
        v8_x2 = self.location[0] + ((v8_x - self.location[0]) * math.cos(self.rotation_y) - (v8_z - self.location[2]) * math.sin(self.rotation_y))
        v8_z2 = self.location[2] + ((v8_z - self.location[2]) * math.cos(self.rotation_y) + (v8_x - self.location[0]) * math.sin(self.rotation_y))
        v8 = (v8_x2, v8_y, v8_z2)

        v9_x = self.location[0]
        v9_y = self.location[1] - self.dimensions[0] / 2
        v9_z = self.location[2] + self.dimensions[2] / 2 + 1
        v9_x2 = self.location[0] + ((v9_x - self.location[0]) * math.cos(self.rotation_y) - (v9_z - self.location[2]) * math.sin(self.rotation_y))
        v9_z2 = self.location[2] + ((v9_z - self.location[2]) * math.cos(self.rotation_y) + (v9_x - self.location[0]) * math.sin(self.rotation_y))
        v9 = (v9_x2, v9_y, v9_z2)

        self.vertices = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)

    def set_location(self,location):
        self.location[0] = location[0]
        self.location[1] = location[1]
        self.location[2] = location[2]
        self.build_vertices()

    def mod_location(self, distances):
        self.set_location((self.location[0] + distances[0], self.location[1] + distances[1],
                           self.location[2] + distances[2]))

    def set_dimensions(self, dimensions):
        self.dimensions[0] = dimensions[0]
        self.dimensions[1] = dimensions[1]
        self.dimensions[2] = dimensions[2]
        self.build_vertices()

    def mod_dimensions(self, distances):
        self.set_dimensions((self.dimensions[0] + distances[0], self.dimensions[1] + distances[1],
                             self.dimensions[2] + distances[2]))

    def set_rotation_y(self, rotation_y):
        self.rotation_y = rotation_y
        self.build_vertices()

    def mod_rotation_y(self, rotation_y):
        self.set_rotation_y(self.rotation_y + rotation_y)

    def round_output(self, x):
        return "%.2f" % round(x,2)

    def to_string(self):
        output = str(self.type) + " " + str(self.truncated) + " " + str(self.occluded) + " "+str(self.alpha) + " "
        for entry in self.bbox:
            output += str(entry) + " "
        for entry in self.dimensions:
            output += str(entry) + " "
        for entry in self.location:
            output += str(entry) + " "
        output += str(self.rotation_y)
        return output

    def to_string_rounded(self):
        output = str(self.type) + " " + str(self.truncated) + " " + str(self.occluded) + " "
        output += str(self.round_output(self.alpha)) + " "
        for entry in self.bbox:
            output += str(self.round_output(entry)) + " "
        for entry in self.dimensions:
            output += str(self.round_output(entry)) + " "
        for entry in self.location:
            output += str(self.round_output(entry)) + " "
        output += str(self.round_output(self.rotation_y))
        return output