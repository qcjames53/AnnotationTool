import math

# Bounding Box Object Class
class BoundingBox():
    """
    A class to represent a 3D box placed along the ground plane at a height of 0
    """
    def __init__(self, object_type, color_value, position=(0,0,0), rotation=0, size=(1,1,1)):
        self.pos = position
        self.pos_init = position
        self.length = size[2]
        self.length_init = self.length
        self.width = size[0]
        self.width_init = self.width
        self.height = size[1]
        self.height_init = self.height
        self.rot = rotation
        self.rot_init = rotation
        self.object_type = object_type
        self.color_value = color_value
        self.vertices = None
        self.build_vertices()

    def build_vertices(self):
        v0_x = self.pos[0] + self.width / 2
        v0_y = self.pos[1]
        v0_z = self.pos[2] - self.length / 2
        v0_x2 = (self.pos[0] + (((v0_x-self.pos[0]) * math.cos(self.rot)) - ((v0_z-self.pos[2]) * math.sin(self.rot))))
        v0_z2 = (self.pos[2] + (((v0_z-self.pos[2]) * math.cos(self.rot)) + ((v0_x-self.pos[0]) * math.sin(self.rot))))
        v0 = (v0_x2,v0_y,v0_z2)

        v1_x = self.pos[0] + self.width / 2
        v1_y = self.pos[1] + self.height
        v1_z = self.pos[2] - self.length / 2
        v1_x2 = self.pos[0] + ((v1_x-self.pos[0]) * math.cos(self.rot) - (v1_z-self.pos[2]) * math.sin(self.rot))
        v1_z2 = self.pos[2] + ((v1_z-self.pos[2]) * math.cos(self.rot) + (v1_x-self.pos[0]) * math.sin(self.rot))
        v1 = (v1_x2, v1_y, v1_z2)

        v2_x = self.pos[0] - self.width / 2
        v2_y = self.pos[1] + self.height
        v2_z = self.pos[2] - self.length / 2
        v2_x2 = self.pos[0] + ((v2_x-self.pos[0]) * math.cos(self.rot) - (v2_z-self.pos[2]) * math.sin(self.rot))
        v2_z2 = self.pos[2] + ((v2_z-self.pos[2]) * math.cos(self.rot) + (v2_x-self.pos[0]) * math.sin(self.rot))
        v2 = (v2_x2, v2_y, v2_z2)

        v3_x = self.pos[0] - self.width / 2
        v3_y = self.pos[1]
        v3_z = self.pos[2] - self.length / 2
        v3_x2 = self.pos[0] + ((v3_x-self.pos[0]) * math.cos(self.rot) - (v3_z-self.pos[2]) * math.sin(self.rot))
        v3_z2 = self.pos[2] + ((v3_z-self.pos[2]) * math.cos(self.rot) + (v3_x-self.pos[0]) * math.sin(self.rot))
        v3 = (v3_x2, v3_y, v3_z2)

        v4_x = self.pos[0] + self.width / 2
        v4_y = self.pos[1]
        v4_z = self.pos[2] + self.length / 2
        v4_x2 = self.pos[0] + ((v4_x-self.pos[0]) * math.cos(self.rot) - (v4_z-self.pos[2]) * math.sin(self.rot))
        v4_z2 = self.pos[2] + ((v4_z-self.pos[2]) * math.cos(self.rot) + (v4_x-self.pos[0]) * math.sin(self.rot))
        v4 = (v4_x2, v4_y, v4_z2)

        v5_x = self.pos[0] + self.width / 2
        v5_y = self.pos[1] + self.height
        v5_z = self.pos[2] + self.length / 2
        v5_x2 = self.pos[0] + ((v5_x-self.pos[0]) * math.cos(self.rot) - (v5_z-self.pos[2]) * math.sin(self.rot))
        v5_z2 = self.pos[2] + ((v5_z-self.pos[2]) * math.cos(self.rot) + (v5_x-self.pos[0]) * math.sin(self.rot))
        v5 = (v5_x2, v5_y, v5_z2)

        v6_x = self.pos[0] - self.width / 2
        v6_y = self.pos[1]
        v6_z = self.pos[2] + self.length / 2
        v6_x2 = self.pos[0] + ((v6_x-self.pos[0]) * math.cos(self.rot) - (v6_z-self.pos[2]) * math.sin(self.rot))
        v6_z2 = self.pos[2] + ((v6_z-self.pos[2]) * math.cos(self.rot) + (v6_x-self.pos[0]) * math.sin(self.rot))
        v6 = (v6_x2, v6_y, v6_z2)

        v7_x = self.pos[0] - self.width / 2
        v7_y = self.pos[1] + self.height
        v7_z = self.pos[2] + self.length / 2
        v7_x2 = self.pos[0] + ((v7_x-self.pos[0]) * math.cos(self.rot) - (v7_z-self.pos[2]) * math.sin(self.rot))
        v7_z2 = self.pos[2] + ((v7_z-self.pos[2]) * math.cos(self.rot) + (v7_x-self.pos[0]) * math.sin(self.rot))
        v7 = (v7_x2, v7_y, v7_z2)

        v8_x = self.pos[0]
        v8_y = self.pos[1] + self.height / 2
        v8_z = self.pos[2] + self.length / 2
        v8_x2 = self.pos[0] + ((v8_x - self.pos[0]) * math.cos(self.rot) - (v8_z - self.pos[2]) * math.sin(self.rot))
        v8_z2 = self.pos[2] + ((v8_z - self.pos[2]) * math.cos(self.rot) + (v8_x - self.pos[0]) * math.sin(self.rot))
        v8 = (v8_x2, v8_y, v8_z2)

        v9_x = self.pos[0]
        v9_y = self.pos[1] + self.height / 2
        v9_z = self.pos[2] + self.length / 2 + 1
        v9_x2 = self.pos[0] + ((v9_x - self.pos[0]) * math.cos(self.rot) - (v9_z - self.pos[2]) * math.sin(self.rot))
        v9_z2 = self.pos[2] + ((v9_z - self.pos[2]) * math.cos(self.rot) + (v9_x - self.pos[0]) * math.sin(self.rot))
        v9 = (v9_x2, v9_y, v9_z2)

        self.vertices = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)

    def reset(self):
        self.pos = self.pos_init
        self.length = self.length_init
        self.width = self.width_init
        self.height = self.height_init
        self.rot = self.rot_init
        self.build_vertices()

    def set_pos(self,pos=(0, 0, 0)):
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]
        self.pos[2] = pos[2]
        self.build_vertices()

    def mod_pos(self,distances=(0, 0, 0)):
        self.pos = (self.pos[0] + distances[0], self.pos[1] + distances[1], self.pos[2] + distances[2])
        self.build_vertices()

    def set_width(self,width):
        self.width = width
        self.build_vertices()

    def mod_width(self,width):
        self.set_width(self.width + width)

    def set_height(self,height):
        self.height = height
        self.build_vertices()

    def mod_height(self,height):
        self.set_height(self.height + height)

    def set_length(self,length):
        self.length = length
        self.build_vertices()

    def mod_length(self,length):
        self.set_length(self.length + length)

    def set_rot(self,rotation):
        self.rot = rotation
        self.build_vertices()

    def mod_rot(self,rotation):
        self.set_rot(self.rot + rotation)

    def round_output(self, x):
        return "%.2f" % round(x,2)

    def to_string(self):
        return "POS:" + self.round_output(self.pos[0]) + "," + self.round_output(self.pos[1]) + "," + self.round_output(self.pos[2]) + "  SIZE:" + self.round_output(self.width) + "," + self.round_output(self.height) + "," + self.round_output(self.length) + " ROT:" + self.round_output(self.rot)
