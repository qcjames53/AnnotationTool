import vpython

#world constants
vpython.scene.autoscale = False
world_vector = vpython.vector(1,0,0)    #a vpython vector pointing straight up in the image world
bounding_box_offset = vpython.vector(0,0,0)
bounding_box_label_offset = vpython.vector(0,0,0)
bounding_box_opacity = 1.0

class bounding_box():
    '''
    Documentation TBD
    '''
    def __init__(self, pos, facing, length, width, height, name, color_value):
        self.name = name
        self.color_value = color_value
        self.pos = pos
        self.facing = facing
        self.outer_box = vpython.box(pos=(self.pos + bounding_box_offset), axis=world_vector,
                                     length=length, height=height, width=width,
                                     color=color_value, opacity=bounding_box_opacity)
        self.label = vpython.label(pos=(self.pos + bounding_box_label_offset), text=name)

    def set_pos(self, pos):
        self.outer_box.pos = pos + bounding_box_offset
        self.label.pos = pos+ bounding_box_label_offset

    def rotate(self, angle):
        self.outer_box.rotate(angle=angle)
        self.label.rotate(angle=angle)


#test code
b = bounding_box(pos=vpython.vector(0,0,0),facing=vpython.vector(0,0,0),length=1,width=4,height=9,name="Monolith",color_value=vpython.vector(1,1,0))

velocity = 0.01
while True:
    vpython.rate(200)
    b.rotate(angle = velocity)