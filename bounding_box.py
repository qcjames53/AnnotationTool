import math

# Bounding Box Object Class
class BoundingBox():
    """
    A class to represent a 3D box placed along the ground plane at a height of 0.

    The following is stolen directly from the Kitti data set:
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
                            #### Note: bounding_box.py will not make use of this,
                            #### type, as we'e dealing with 3D.
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.

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
        v0_y = self.location[1] + self.dimensions[0]
        v0_z = self.location[2] - self.dimensions[2] / 2
        v0_x2 = (self.location[0] + (((v0_x-self.location[0]) * math.cos(self.rotation_y)) - ((v0_z-self.location[2]) * math.sin(self.rotation_y))))
        v0_z2 = (self.location[2] + (((v0_z-self.location[2]) * math.cos(self.rotation_y)) + ((v0_x-self.location[0]) * math.sin(self.rotation_y))))
        v0 = (v0_x2,v0_y,v0_z2)

        v1_x = self.location[0] + self.dimensions[1] / 2
        v1_y = self.location[1]
        v1_z = self.location[2] - self.dimensions[2] / 2
        v1_x2 = self.location[0] + ((v1_x-self.location[0]) * math.cos(self.rotation_y) - (v1_z-self.location[2]) * math.sin(self.rotation_y))
        v1_z2 = self.location[2] + ((v1_z-self.location[2]) * math.cos(self.rotation_y) + (v1_x-self.location[0]) * math.sin(self.rotation_y))
        v1 = (v1_x2, v1_y, v1_z2)

        v2_x = self.location[0] - self.dimensions[1] / 2
        v2_y = self.location[1]
        v2_z = self.location[2] - self.dimensions[2] / 2
        v2_x2 = self.location[0] + ((v2_x-self.location[0]) * math.cos(self.rotation_y) - (v2_z-self.location[2]) * math.sin(self.rotation_y))
        v2_z2 = self.location[2] + ((v2_z-self.location[2]) * math.cos(self.rotation_y) + (v2_x-self.location[0]) * math.sin(self.rotation_y))
        v2 = (v2_x2, v2_y, v2_z2)

        v3_x = self.location[0] - self.dimensions[1] / 2
        v3_y = self.location[1] + self.dimensions[0]
        v3_z = self.location[2] - self.dimensions[2] / 2
        v3_x2 = self.location[0] + ((v3_x-self.location[0]) * math.cos(self.rotation_y) - (v3_z-self.location[2]) * math.sin(self.rotation_y))
        v3_z2 = self.location[2] + ((v3_z-self.location[2]) * math.cos(self.rotation_y) + (v3_x-self.location[0]) * math.sin(self.rotation_y))
        v3 = (v3_x2, v3_y, v3_z2)

        v4_x = self.location[0] + self.dimensions[1] / 2
        v4_y = self.location[1] + self.dimensions[0]
        v4_z = self.location[2] + self.dimensions[2] / 2
        v4_x2 = self.location[0] + ((v4_x-self.location[0]) * math.cos(self.rotation_y) - (v4_z-self.location[2]) * math.sin(self.rotation_y))
        v4_z2 = self.location[2] + ((v4_z-self.location[2]) * math.cos(self.rotation_y) + (v4_x-self.location[0]) * math.sin(self.rotation_y))
        v4 = (v4_x2, v4_y, v4_z2)

        v5_x = self.location[0] + self.dimensions[1] / 2
        v5_y = self.location[1]
        v5_z = self.location[2] + self.dimensions[2] / 2
        v5_x2 = self.location[0] + ((v5_x-self.location[0]) * math.cos(self.rotation_y) - (v5_z-self.location[2]) * math.sin(self.rotation_y))
        v5_z2 = self.location[2] + ((v5_z-self.location[2]) * math.cos(self.rotation_y) + (v5_x-self.location[0]) * math.sin(self.rotation_y))
        v5 = (v5_x2, v5_y, v5_z2)

        v6_x = self.location[0] - self.dimensions[1] / 2
        v6_y = self.location[1] + self.dimensions[0]
        v6_z = self.location[2] + self.dimensions[2] / 2
        v6_x2 = self.location[0] + ((v6_x-self.location[0]) * math.cos(self.rotation_y) - (v6_z-self.location[2]) * math.sin(self.rotation_y))
        v6_z2 = self.location[2] + ((v6_z-self.location[2]) * math.cos(self.rotation_y) + (v6_x-self.location[0]) * math.sin(self.rotation_y))
        v6 = (v6_x2, v6_y, v6_z2)

        v7_x = self.location[0] - self.dimensions[1] / 2
        v7_y = self.location[1]
        v7_z = self.location[2] + self.dimensions[2] / 2
        v7_x2 = self.location[0] + ((v7_x-self.location[0]) * math.cos(self.rotation_y) - (v7_z-self.location[2]) * math.sin(self.rotation_y))
        v7_z2 = self.location[2] + ((v7_z-self.location[2]) * math.cos(self.rotation_y) + (v7_x-self.location[0]) * math.sin(self.rotation_y))
        v7 = (v7_x2, v7_y, v7_z2)

        v8_x = self.location[0]
        v8_y = self.location[1] + self.dimensions[0] / 2
        v8_z = self.location[2] + self.dimensions[2] / 2
        v8_x2 = self.location[0] + ((v8_x - self.location[0]) * math.cos(self.rotation_y) - (v8_z - self.location[2]) * math.sin(self.rotation_y))
        v8_z2 = self.location[2] + ((v8_z - self.location[2]) * math.cos(self.rotation_y) + (v8_x - self.location[0]) * math.sin(self.rotation_y))
        v8 = (v8_x2, v8_y, v8_z2)

        v9_x = self.location[0]
        v9_y = self.location[1] + self.dimensions[0] / 2
        v9_z = self.location[2] + self.dimensions[2] / 2 + 1
        v9_x2 = self.location[0] + ((v9_x - self.location[0]) * math.cos(self.rotation_y) - (v9_z - self.location[2]) * math.sin(self.rotation_y))
        v9_z2 = self.location[2] + ((v9_z - self.location[2]) * math.cos(self.rotation_y) + (v9_x - self.location[0]) * math.sin(self.rotation_y))
        v9 = (v9_x2, v9_y, v9_z2)

        self.vertices = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)


    def build_alpha(self, cam_location):
        delta_x = self.location[0] - cam_location[0]
        delta_z = self.location[2] - cam_location[2]

        r1 = (self.rotation_y) % (2 * math.pi)
        r2 = (math.atan2(delta_z, delta_x) - (1.5 * math.pi)) % (2 * math.pi)

        self.alpha = (r1 - r2) % (2 * math.pi)

    def set_truncation(self, truncation):
        self.truncated = truncation

    def set_occlusion(self, occlusion):
        self.occluded = occlusion

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
        self.set_dimensions((max(self.dimensions[0] + distances[0],0), max(self.dimensions[1] + distances[1],0),
                             max(self.dimensions[2] + distances[2],0)))

    def set_rotation_y(self, rotation_y):
        rotation_y %= (2 * math.pi)
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