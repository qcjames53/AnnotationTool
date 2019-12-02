import vpython
import math
import random
from enum import Enum

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

bounding_box_label_offset = vpython.vector(5, 10, 0)
bounding_box_opacity = 0.25
bounding_box_selected_opacity = 0.5
bounding_box_corner_size = 1
bounding_box_corner_color = vpython.color.red
boxes = [None] #contains None to start at index 1
selected_bounding_box = -1


class MouseState(Enum):
    IDLE = 0
    TRANSLATE = 1
    HEIGHT = 2
    WIDTH = 3
    DEPTH = 4


mouseState = MouseState.IDLE


class BoundingBox:
    """
    A bounding box object to be instantiated around wanted physical objects in a scene.
    """
    def __init__(self, pos, rotation, length, width, height, index, color_value, object_type):
        self.index = index
        self.object_type = object_type
        self.pos = pos
        self.length = length
        self.width = width
        self.height = height
        self.center = vpython.vector(self.pos.x - self.length / 2, self.pos.y + self.height / 2, self.pos.z + self.width / 2)
        self.rotation = 0
        self.s_key = float("0.00"+str(index))
        self.outer_box = vpython.box(pos=self.center,
                                     length=self.length, height=self.height, width=self.width,
                                     color=color_value, opacity=bounding_box_opacity, pickable=True, shininess=self.s_key)
        self.label = vpython.label(pos=self.center, text=object_type + " " + str(index),
                                   xoffset=bounding_box_label_offset.x, yoffset=bounding_box_label_offset.y,
                                   pickable=False)
        self.rotate(rotation)

    def set_opacity(self, value):
        self.outer_box.opacity = value

    def translate(self, vector):
        self.center += vector

        self.outer_box.pos = self.center
        self.label.pos = self.center

    def set_pos(self, pos):
        self.pos = pos
        self.center = vpython.vector(self.pos.x - self.length / 2, self.pos.y + self.height / 2, self.pos.z + self.width / 2)
        self.outer_box.pos = self.center
        self.label.pos = self.center

    def set_width(self,width):
        self.width = width
        self.outer_box.width = self.width
        self.set_pos(self.pos)

    def set_height(self, height):
        self.height = height
        self.outer_box.height = self.height
        self.set_pos(self.pos)

    def set_length(self, length):
        self.length = length
        self.outer_box.length = self.length
        self.set_pos(self.pos)

    def rotate(self, degrees):
        self.rotation += degrees
        self.outer_box.rotate(angle=vpython.radians(degrees), axis=vpython.vector(0,1,0), origin=self.center)
        self.label.rotate(angle=vpython.radians(degrees), axis=vpython.vector(0, 1, 0), origin=self.center)

    def rotate_to(self, yaw):
        self.rotate(degrees=(yaw - self.rotation))
#--end of class--

def instantiate_box(pos=vpython.vector(0,0,0), rotation=0, length=4, width=8, height=4, object_type="Car"):
    color_value = vpython.vector(random.random(), random.random(), random.random())
    box = BoundingBox(pos=pos, rotation=rotation, length=length, width=width, height=height, object_type=object_type, color_value=color_value, index=len(boxes))
    boxes.append(box)

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
    if(selected_bounding_box > 1):
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


def mouse_position_ground():
    return scene.mouse.project(normal=vpython.vector(0,1,0))


def mouse_down():
    global mouseState
    global selected_bounding_box
    obj = scene.mouse.pick
    if obj is None:
        instantiate_box(pos=mouse_position_ground())
    else:
        selected_bounding_box = int(str(obj.shininess)[4:])
        if scene.mouse.ctrl:
            mouseState = MouseState.WIDTH
        elif scene.mouse.alt:
            mouseState = MouseState.DEPTH
        elif scene.mouse.shift:
            mouseState = MouseState.HEIGHT
        else:
            mouseState = MouseState.TRANSLATE


def mouse_up():
    global mouseState
    global selected_bounding_box
    mouseState = MouseState.IDLE
    selected_bounding_box = -1

def mouse_move():
    global mouseState
    global selected_bounding_box
    if mouseState == MouseState.IDLE or selected_bounding_box == -1:
        return
    elif mouseState == MouseState.TRANSLATE:
        boxes[selected_bounding_box].set_pos(mouse_position_ground())
    elif mouseState == MouseState.WIDTH:
        boxes[selected_bounding_box].set_length(vpython.mag(boxes[selected_bounding_box].pos - mouse_position_ground()))
    elif mouseState == MouseState.HEIGHT:
        boxes[selected_bounding_box].set_height(vpython.mag(boxes[selected_bounding_box].pos - mouse_position_ground()))
    elif mouseState == MouseState.DEPTH:
        boxes[selected_bounding_box].set_width(vpython.mag(boxes[selected_bounding_box].pos - mouse_position_ground()))


# world calibration view
x_axis = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(1,0,0), color=vpython.vector(1,0,0), shaftwidth=0.1, visible=False, pickable=False)
y_axis = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(0,1,0), color=vpython.vector(0,1,0), shaftwidth=0.1, visible=False, pickable=False)
z_axis = vpython.arrow(pos=vpython.vector(-10,5,0), axis=vpython.vector(0,0,1), color=vpython.vector(0,0,1), shaftwidth=0.1, visible=False, pickable=False)
alignment_grid_1 = vpython.arrow(pos=vpython.vector(20,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_2 = vpython.arrow(pos=vpython.vector(16,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_3 = vpython.arrow(pos=vpython.vector(12,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_4 = vpython.arrow(pos=vpython.vector(8,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_5 = vpython.arrow(pos=vpython.vector(4,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_6 = vpython.arrow(pos=vpython.vector(0,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_7 = vpython.arrow(pos=vpython.vector(-4,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_8 = vpython.arrow(pos=vpython.vector(-8,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_9 = vpython.arrow(pos=vpython.vector(-12,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_10 = vpython.arrow(pos=vpython.vector(-16,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
alignment_grid_11 = vpython.arrow(pos=vpython.vector(-20,0,0), axis=vpython.vector(0,0,20), color=vpython.vector(0,0,1), shaftwidth=0.1, headwidth=0.1, visible=False, pickable=False)
label_camera_pitch = vpython.label(pixel_pos=True, align="left", pos=vpython.vector(10,image_resolution[1]-15,0), text="Camera pitch: ", visible=False, pickable=False)
label_camera_fov = vpython.label(pixel_pos=True, align="left", pos=vpython.vector(10,image_resolution[1]-40,0), text="Camera fov: ", visible=False, pickable=False)

# camera image display
image = vpython.box(axis=vpython.vector(-1,0,0), shininess=(0), emissive=True, texture={'file':'/test_image.png'}, height = 1, pickable=False)
draw_image()

#bounding box selection display
label_box_selection = vpython.label(pixel_pos=True, align="center", pos=vpython.vector(image_resolution[0]/2,image_resolution[1]-15,0), text="Selected Bounding Box: 0", visible=True, pickable=False)

#GUI controls
scene.bind('mousedown', mouse_down)
scene.bind('mouseup', mouse_up)
scene.bind('mousemove', mouse_move)

#GUI elements
scene.append_to_caption("\nBounding Box Controls:\n")
vpython.button(bind=instantiate_box, text="New")
vpython.button(bind=set_bounding_box_n_1, text="Select Prev")
vpython.button(bind=set_bounding_box_1, text="Select Next")

scene.append_to_caption("\n")
vpython.button(bind=move_box_x, text="+x")
vpython.button(bind=move_box_n_x, text="-x")
vpython.button(bind=move_box_z, text="+z")
vpython.button(bind=move_box_n_z, text="-z")
vpython.button(bind=rotate_box_1, text="+rot")
vpython.button(bind=rotate_box_n_1, text="-rot")

scene.append_to_caption("\n\nCamera Controls\n")
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