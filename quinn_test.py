import vpython
import math

# world constants
image_resolution = (720,480)

scene = vpython.canvas(title='Cube Draw Test', width=image_resolution[0], height=image_resolution[1],)
scene.autoscale = False
scene.userpan = False
scene.userspin = True
scene.userzoom = True
scene.range = 10

bounding_box_offset = vpython.vector(0, 0, 0)
bounding_box_label_offset = vpython.vector(0, 0, 0)
bounding_box_opacity = 1.0


class BoundingBox:
    '''
    Documentation TBD
    '''
    def __init__(self, pos, facing, length, width, height, name, color_value):
        self.name = name
        self.color_value = color_value
        self.pos = pos
        self.facing = facing
        self.outer_box = vpython.box(pos=(self.pos + bounding_box_offset),
                                     length=length, height=height, width=width,
                                     color=color_value, opacity=bounding_box_opacity)
        self.label = vpython.label(pos=(self.pos + bounding_box_label_offset), text=name, xoffset=20, yoffset=50)

    def set_pos(self, pos):
        self.outer_box.pos = pos + bounding_box_offset
        self.label.pos = pos + bounding_box_label_offset

    def rotate(self, angle):
        self.outer_box.rotate(angle=angle)
        self.label.rotate(angle=angle)

def draw_image(a,b):
    image_height = 4 * math.tan(scene.fov / 2) * vpython.mag(scene.center - scene.camera.pos)
    image.pos = scene.camera.pos * -1
    image.up = scene.forward * -1
    image.width = image_height
    image.length=image_height * (image_resolution[0] / image_resolution[1])
    if(a != -1):
        alabel1.text = "Camera rotation: " + str(a)
    if (b != -1):
        alabel2.text = "Camera FOV: " + str(b)


def change_angle(s):
    scene.forward = vpython.vector(0,0,-1)
    scene.forward = scene.forward.rotate(angle=vpython.radians(s.value), axis=vpython.vector(1, 0, 0))
    draw_image(s.value,-1)


def change_fov(s):
    scene.fov = s.value
    draw_image(-1,s.value)

# world calibration view
a1 = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(1,0,0), color=vpython.vector(1,0,0), shaftwidth=0.1)
a2 = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(0,1,0), color=vpython.vector(0,1,0), shaftwidth=0.1)
a3 = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(0,0,1), color=vpython.vector(0,0,1), shaftwidth=0.1)
a4 = vpython.arrow(pos=vpython.vector(20,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a5 = vpython.arrow(pos=vpython.vector(16,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a6 = vpython.arrow(pos=vpython.vector(12,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a7 = vpython.arrow(pos=vpython.vector(8,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a8 = vpython.arrow(pos=vpython.vector(4,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a9 = vpython.arrow(pos=vpython.vector(0,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a10 = vpython.arrow(pos=vpython.vector(-4,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a11 = vpython.arrow(pos=vpython.vector(-8,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a12 = vpython.arrow(pos=vpython.vector(-12,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a13 = vpython.arrow(pos=vpython.vector(-16,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
a14 = vpython.arrow(pos=vpython.vector(-20,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1)
alabel1 = vpython.label(pixel_pos=True, align="left", pos=vpython.vector(10,image_resolution[1]-15,0))
alabel2 = vpython.label(pixel_pos=True, align="left", pos=vpython.vector(10,image_resolution[1]-40,0))

# camera image display
image = vpython.box(axis=vpython.vector(-1,0,0), shininess=(0), emissive=True, texture={'file':'/test_image.png'}, height = 1)
draw_image(-1,-1)

scene.append_to_caption("\nChange camera angle")
vpython.slider(bind=change_angle, min=180, max=270, step=1, value=180)
scene.append_to_caption('\n\n')

scene.append_to_caption("Change fov")
vpython.slider(bind=change_fov, min=0, max=math.pi, value=0 )
scene.append_to_caption('\n\n')

#b = BoundingBox(pos=vpython.vector(0,0,1),facing=vpython.vector(1,0,0),length=1,width=1,height=1,name="Test Name",color_value=vpython.vector(1,0,1))