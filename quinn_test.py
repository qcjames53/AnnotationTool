import vpython
import math
import random

# world constants
image_resolution = (720,480)

scene = vpython.canvas(width=image_resolution[0], height=image_resolution[1],)
scene.autoscale = False
scene.userpan = False
scene.userspin = False
scene.userzoom = False
scene.range = 10

camera_pitch_stored = 0
camera_fov_stored = 0
initial_camera_pitch = 210
initial_camera_fov = 39

bounding_box_label_offset = vpython.vector(0, 0, 0)
bounding_box_opacity = 0.25
bounding_box_selected_opacity = 0.5
boxes = []
selected_bounding_box = 0

class BoundingBox:
    '''
    Documentation TBD
    '''
    def __init__(self, pos, rotation, length, width, height, index, color_value, object_type):
        self.index = index
        self.object_type = object_type
        self.rotation = rotation
        self.outer_box = vpython.box(pos=(pos),
                                     length=length, height=height, width=width,
                                     color=color_value, opacity=bounding_box_opacity)
        self.label = vpython.label(pos=(self.outer_box.pos + bounding_box_label_offset), text=object_type + " " + str(index), xoffset=20, yoffset=50)

    def set_opacity(self, value):
        self.outer_box.opacity = value

    def set_pos(self, pos):
        self.outer_box.pos = pos
        self.label.pos = self.outer_box.pos + bounding_box_label_offset

    def translate(self, vector):
        self.outer_box.pos += vector
        self.label.pos = self.outer_box.pos + bounding_box_label_offset + vector

    def rotate(self, degrees):
        self.rotation += degrees
        self.outer_box.rotate(angle=vpython.radians(degrees), axis=vpython.vector(0,1,0), origin=self.outer_box.pos)
        self.label.rotate(angle=vpython.radians(degrees), axis=vpython.vector(0, 1, 0), origin=self.outer_box.pos)

    def rotate_to(self, yaw):
        self.rotate(degrees=(yaw - self.rotation))
#--end of class--

def instantiate_box(pos, rotation, length, width, height, object_type, color_value):
    box = BoundingBox(pos=pos, rotation=rotation, length=length, width=width, height=height, object_type=object_type, color_value=color_value, index=len(boxes))
    boxes.append(box)

def instantiate_box_auto():
    temp_color = vpython.vector(random.random(),random.random(),random.random())
    instantiate_box(pos=vpython.vector(0,0,0), rotation=0, length=5, width=10, height=5, color_value=temp_color, object_type="Car")

def draw_image():
    image_height = 4 * math.tan(scene.fov / 2) * vpython.mag(scene.center - scene.camera.pos)
    image.pos = scene.camera.pos * -1
    image.up = scene.forward * -1
    image.width = image_height
    image.length=image_height * (image_resolution[0] / image_resolution[1])
    label_camera_pitch.text = "Camera pitch: " + str(camera_pitch_stored)
    label_camera_fov.text = "Camera fov: " + str(camera_fov_stored)


def enable_calibration_overlay(b):
    x_axis.visible = b.checked
    y_axis.visible = b.checked
    z_axis.visible = b.checked
    alignment_grid_1.visible = b.checked
    alignment_grid_2.visible = b.checked
    alignment_grid_3.visible = b.checked
    alignment_grid_4.visible = b.checked
    alignment_grid_5.visible = b.checked
    alignment_grid_6.visible = b.checked
    alignment_grid_7.visible = b.checked
    alignment_grid_8.visible = b.checked
    alignment_grid_9.visible = b.checked
    alignment_grid_10.visible = b.checked
    alignment_grid_11.visible = b.checked
    label_camera_pitch.visible = b.checked
    label_camera_fov.visible = b.checked


def set_camera_pitch_n_10():
    set_camera_pitch(-10)
def set_camera_pitch_n_1():
    set_camera_pitch(-1)
def set_camera_pitch_1():
    set_camera_pitch(1)
def set_camera_pitch_10():
    set_camera_pitch(10)
def set_camera_pitch(s):
    global camera_pitch_stored
    camera_pitch_stored += s
    scene.forward = scene.forward.rotate(angle=vpython.radians(s), axis=vpython.vector(1, 0, 0))
    draw_image()

def set_camera_fov_n_10():
    set_camera_fov(-10)
def set_camera_fov_n_1():
    set_camera_fov(-1)
def set_camera_fov_1():
    set_camera_fov(1)
def set_camera_fov_10():
    set_camera_fov(10)
def set_camera_fov(s):
    global camera_fov_stored
    camera_fov_stored += s
    scene.fov = vpython.radians(camera_fov_stored)
    draw_image()


def set_bounding_box_n_1():
    if(selected_bounding_box > 0):
        set_bounding_box(-1)
def set_bounding_box_1():
    if(selected_bounding_box < len(boxes)-1):
        set_bounding_box(1)
def set_bounding_box(n):
    global selected_bounding_box
    boxes[selected_bounding_box].set_opacity(bounding_box_opacity)
    selected_bounding_box += n
    label_box_selection.text = "Selected Bounding Box: " + str(selected_bounding_box)
    boxes[selected_bounding_box].set_opacity(bounding_box_selected_opacity)

def move_box_n_x():
    move_box(x=-1,z=0)
def move_box_x():
    move_box(x=1, z=0)
def move_box_n_z():
    move_box(x=0, z=-1)
def move_box_z():
    move_box(x=0, z=1)
def move_box(x, z):
    boxes[selected_bounding_box].translate(vpython.vector(x,0,z))


def rotate_box_n_1():
    rotate_box(-1)
def rotate_box_1():
    rotate_box(1)
def rotate_box(deg):
    boxes[selected_bounding_box].rotate(deg)

# world calibration view
x_axis = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(1,0,0), color=vpython.vector(1,0,0), shaftwidth=0.1, visible=False)
y_axis = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(0,1,0), color=vpython.vector(0,1,0), shaftwidth=0.1, visible=False)
z_axis = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(0,0,1), color=vpython.vector(0,0,1), shaftwidth=0.1, visible=False)
alignment_grid_1 = vpython.arrow(pos=vpython.vector(20,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_2 = vpython.arrow(pos=vpython.vector(16,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_3 = vpython.arrow(pos=vpython.vector(12,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_4 = vpython.arrow(pos=vpython.vector(8,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_5 = vpython.arrow(pos=vpython.vector(4,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_6 = vpython.arrow(pos=vpython.vector(0,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_7 = vpython.arrow(pos=vpython.vector(-4,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_8 = vpython.arrow(pos=vpython.vector(-8,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_9 = vpython.arrow(pos=vpython.vector(-12,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_10 = vpython.arrow(pos=vpython.vector(-16,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
alignment_grid_11 = vpython.arrow(pos=vpython.vector(-20,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False)
label_camera_pitch = vpython.label(pixel_pos=True, align="left", pos=vpython.vector(10,image_resolution[1]-15,0), text="Camera pitch: ", visible=False)
label_camera_fov = vpython.label(pixel_pos=True, align="left", pos=vpython.vector(10,image_resolution[1]-40,0), text="Camera fov: ", visible=False)

# camera image display
image = vpython.box(axis=vpython.vector(-1,0,0), shininess=(0), emissive=True, texture={'file':'/test_image.png'}, height = 1)
draw_image()

#bounding box selection display
label_box_selection = vpython.label(pixel_pos=True, align="center", pos=vpython.vector(image_resolution[0]/2,image_resolution[1]-15,0), text="Selected BB: ", visible=True)

#GUI elements
scene.append_to_caption("\nBounding Box Controls:\n")
vpython.button(bind=instantiate_box_auto, text="New")
vpython.button(bind=set_bounding_box_n_1, text="Select Prev")
vpython.button(bind=set_bounding_box_1, text="Select Next")

scene.append_to_caption("\n")
vpython.button(bind=move_box_x, text="+x")
vpython.button(bind=move_box_n_x, text="-x")
vpython.button(bind=move_box_z, text="+z")
vpython.button(bind=move_box_n_z, text="-z")
vpython.button(bind=rotate_box_1, text="+rot")
vpython.button(bind=rotate_box_n_1, text="-rot")

scene.append_to_caption("\n\nCamera Controls")
vpython.checkbox(bind=enable_calibration_overlay, text='Enable Calibration Overlay')

scene.append_to_caption("\nSet camera pitch")
vpython.button(bind=set_camera_pitch_n_10, text="-10")
vpython.button(bind=set_camera_pitch_n_1, text="-1")
vpython.button(bind=set_camera_pitch_1, text="+1")
vpython.button(bind=set_camera_pitch_10, text="+10")

scene.append_to_caption("\nSet camera fov")
vpython.button(bind=set_camera_fov_n_10, text="-10")
vpython.button(bind=set_camera_fov_n_1, text="-1")
vpython.button(bind=set_camera_fov_1, text="+1")
vpython.button(bind=set_camera_fov_10, text="+10")

#-----------------begin scene start code---------------------------
set_camera_pitch(initial_camera_pitch)
set_camera_fov(initial_camera_fov)

#b = BoundingBox(pos=vpython.vector(0,0,1),facing=vpython.vector(1,0,0),length=5,width=1,height=1,name="Test Name",color_value=vpython.vector(1,0,1))