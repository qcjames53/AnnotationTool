import random
import numpy
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2

# Constants
camera_rot = (30, 0)
box_translation_amount = 0.1
box_rotation_amount = 2
box_mod_dimension_amount = 0.1
box_mod_ctrl_multiplier = 10
box_blink_speed = 15
box_blink_frame = 0
box_blink_state = False
edge_render_order = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

# Global Variables for PyGame & Data Storage
boxes = []
selected_box = 0
show_ground_plane_grid = False

# Initialize Pygame Display Window & OpenGL
pygame.init()
render_size = (720, 480)
render = pygame.display.set_mode(render_size, DOUBLEBUF | OPENGL)  # Double buffer for monitor refresh rate & OpenGL support in Pygame
gluPerspective(45, (render_size[0] / render_size[1]), 0.1, 50.0)  # (FOV, Aspect Ratio, Near Clipping Plane, Far Clipping Plane)
glTranslatef(0.0,0.0, -5)  # move camera back 5 units
pygame.display.set_caption("Open GL Test")
clock = pygame.time.Clock()

# Bounding Box Object Class
class BoundingBox():
    """
    Documentation is TBD
    """
    def __init__(self, index, object_type, color_value, position=(0,0,0), rotation=0, width=1.5, height=1, length=2):
        self.index = index
        self.pos = position
        self.pos_init = position
        self.length = length
        self.length_init = length
        self.width = width
        self.width_init = width
        self.height = height
        self.height_init = height
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
        v0_x = self.pos[0] + ((v0_x-self.pos[0]) * math.cos(self.rot) - (v0_z-self.pos[2]) * math.sin(self.rot))
        v0_z = self.pos[2] + ((v0_z-self.pos[2]) * math.cos(self.rot) + (v0_x-self.pos[0]) * math.sin(self.rot))
        v0 = (v0_x,v0_y,v0_z)

        v1_x = self.pos[0] + self.width / 2
        v1_y = self.pos[1] + self.height
        v1_z = self.pos[2] - self.length / 2
        v1_x = self.pos[0] + ((v1_x-self.pos[0]) * math.cos(self.rot) - (v1_z-self.pos[2]) * math.sin(self.rot))
        v1_z = self.pos[2] + ((v1_z-self.pos[2]) * math.cos(self.rot) + (v1_x-self.pos[0]) * math.sin(self.rot))
        v1 = (v1_x, v1_y, v1_z)

        v2_x = self.pos[0] - self.width / 2
        v2_y = self.pos[1] + self.height
        v2_z = self.pos[2] - self.length / 2
        v2_x = self.pos[0] + ((v2_x-self.pos[0]) * math.cos(self.rot) - (v2_z-self.pos[2]) * math.sin(self.rot))
        v2_z = self.pos[2] + ((v2_z-self.pos[2]) * math.cos(self.rot) + (v2_x-self.pos[0]) * math.sin(self.rot))
        v2 = (v2_x, v2_y, v2_z)

        v3_x = self.pos[0] - self.width / 2
        v3_y = self.pos[1]
        v3_z = self.pos[2] - self.length / 2
        v3_x = self.pos[0] + ((v3_x-self.pos[0]) * math.cos(self.rot) - (v3_z-self.pos[2]) * math.sin(self.rot))
        v3_z = self.pos[2] + ((v3_z-self.pos[2]) * math.cos(self.rot) + (v3_x-self.pos[0]) * math.sin(self.rot))
        v3 = (v3_x, v3_y, v3_z)

        v4_x = self.pos[0] + self.width / 2
        v4_y = self.pos[1]
        v4_z = self.pos[2] + self.length / 2
        v4_x = self.pos[0] + ((v4_x-self.pos[0]) * math.cos(self.rot) - (v4_z-self.pos[2]) * math.sin(self.rot))
        v4_z = self.pos[2] + ((v4_z-self.pos[2]) * math.cos(self.rot) + (v4_x-self.pos[0]) * math.sin(self.rot))
        v4 = (v4_x, v4_y, v4_z)

        v5_x = self.pos[0] + self.width / 2
        v5_y = self.pos[1] + self.height
        v5_z = self.pos[2] + self.length / 2
        v5_x = self.pos[0] + ((v5_x-self.pos[0]) * math.cos(self.rot) - (v5_z-self.pos[2]) * math.sin(self.rot))
        v5_z = self.pos[2] + ((v5_z-self.pos[2]) * math.cos(self.rot) + (v5_x-self.pos[0]) * math.sin(self.rot))
        v5 = (v5_x, v5_y, v5_z)

        v6_x = self.pos[0] - self.width / 2
        v6_y = self.pos[1]
        v6_z = self.pos[2] + self.length / 2
        v6_x = self.pos[0] + ((v6_x-self.pos[0]) * math.cos(self.rot) - (v6_z-self.pos[2]) * math.sin(self.rot))
        v6_z = self.pos[2] + ((v6_z-self.pos[2]) * math.cos(self.rot) + (v6_x-self.pos[0]) * math.sin(self.rot))
        v6 = (v6_x, v6_y, v6_z)

        v7_x = self.pos[0] - self.width / 2
        v7_y = self.pos[1] + self.height
        v7_z = self.pos[2] + self.length / 2
        v7_x = self.pos[0] + ((v7_x-self.pos[0]) * math.cos(self.rot) - (v7_z-self.pos[2]) * math.sin(self.rot))
        v7_z = self.pos[2] + ((v7_z-self.pos[2]) * math.cos(self.rot) + (v7_x-self.pos[0]) * math.sin(self.rot))
        v7 = (v7_x, v7_y, v7_z)

        self.vertices = (v0, v1, v2, v3, v4, v5, v6, v7)

        # self.vertices = (
        #     (self.pos[0] + self.width / 2, self.pos[1], self.pos[2] - self.length / 2),
        #     (self.pos[0] + self.width / 2, self.pos[1] + self.height, self.pos[2] - self.length / 2),
        #     (self.pos[0] - self.width / 2, self.pos[1] + self.height, self.pos[2] - self.length / 2),
        #     (self.pos[0] - self.width / 2, self.pos[1], self.pos[2] - self.length / 2),
        #     (self.pos[0] + self.width / 2, self.pos[1], self.pos[2] + self.length / 2),
        #     (self.pos[0] + self.width / 2, self.pos[1] + self.height, self.pos[2] + self.length / 2),
        #     (self.pos[0] - self.width / 2, self.pos[1], self.pos[2] + self.length / 2),
        #     (self.pos[0] - self.width / 2, self.pos[1] + self.height, self.pos[2] + self.length / 2)
        # )

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


def draw_bounding_box(index, selected=False):
    glBegin(GL_LINES)
    if selected:
        glColor3fv((1, 1, 1))
    else:
        glColor3fv(boxes[index].color_value)
    for edge in edge_render_order:
        for vertex in edge:
            glVertex3fv(boxes[index].vertices[vertex])
    glEnd()


def draw_axis():
    glBegin(GL_LINES)
    glColor3fv((1, 0, 0))
    glVertex3fv((0, 0, 0))
    glVertex3fv((1, 0, 0))
    glColor3fv((0, 1, 0))
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 1, 0))
    glColor3fv((0, 0, 1))
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 0, 1))
    glEnd()


def draw_ground_plane_grid(lines, distance_between_lines):
    distance = lines * distance_between_lines
    plane_max = distance / 2
    plane_min = -plane_max
    glBegin(GL_LINES)
    glColor3fv((0,0,1))
    for i in range(0,lines+1):
        glVertex3fv((plane_min + i*distance_between_lines,0,plane_min))
        glVertex3fv((plane_min + i*distance_between_lines,0,plane_max))
        glVertex3fv((plane_min, 0, plane_min + i*distance_between_lines))
        glVertex3fv((plane_max, 0, plane_min + i*distance_between_lines))
    glEnd()


def draw_background_image():
    None


def instantiate_box(position=(0,0,0), rotation=0, width=1.5, height=1, length=2, object_type="Car"):
    color_value = (random.random(), random.random(), random.random())
    new_box = BoundingBox(position=position, rotation=rotation, length=length, width=width, height=height,
                      object_type=object_type, color_value=color_value, index=len(boxes))
    global selected_box
    selected_box = new_box.index
    boxes.append(new_box)


# load_texture()
glRotatef(camera_rot[0], 1, 0, 0)  # rotation (angle, x, y, z)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Create New Box
                instantiate_box()
            if (event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE) and len(boxes) > 0:  # Delete Selected Box
                del boxes[selected_box]
                selected_box = min(selected_box,len(boxes)-1)  # clamp selected_box to usable range
            if event.key == pygame.K_PAGEDOWN and selected_box > 0 :  # Select Previous Box
                selected_box -= 1
            if event.key == pygame.K_PAGEUP and selected_box < len(boxes)-1:  # Select Next Box
                selected_box += 1
            if pygame.key.get_mods() & pygame.KMOD_ALT:  # Toggle Ground Plane Grid Visibility
                show_ground_plane_grid = not show_ground_plane_grid
            if event.key == pygame.K_SPACE:  # Reset Selected Box
                boxes[selected_box].reset()
            if event.key == pygame.K_UP:  # Translate Selected Box
                boxes[selected_box].mod_pos((0, 0, -box_translation_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
            if event.key == pygame.K_DOWN:
                boxes[selected_box].mod_pos((0, 0, box_translation_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
            if event.key == pygame.K_LEFT:
                boxes[selected_box].mod_pos((-box_translation_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
            if event.key == pygame.K_RIGHT:
                boxes[selected_box].mod_pos((box_translation_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
            if event.key == pygame.K_w:  # Adjust Height of Selected Box
                boxes[selected_box].mod_height(box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_s and boxes[selected_box].height - (box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:
                boxes[selected_box].mod_height(-box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_a and boxes[selected_box].width - (box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:  # Adjust Width of Selected Box
                boxes[selected_box].mod_width(-box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_d:
                boxes[selected_box].mod_width(box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_q and boxes[selected_box].length - (box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:  # Adjust Length / Depth of Selected Box
                boxes[selected_box].mod_length(-box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_e:
                boxes[selected_box].mod_length(box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_r:  # Adjust Rotation of Selected Box
                boxes[selected_box].mod_rot(box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            if event.key == pygame.K_f:
                boxes[selected_box].mod_rot(-box_mod_dimension_amount * (box_mod_ctrl_multiplier if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#   draw_background_image()
    if show_ground_plane_grid:
        draw_ground_plane_grid(100, 0.5)
        draw_axis()

    for box in boxes:
        if selected_box == box.index and box_blink_state:
            draw_bounding_box(box.index, True)
        else:
            draw_bounding_box(box.index)

    pygame.display.flip()
    box_blink_frame += 1
    if box_blink_frame >= box_blink_speed:
        box_blink_state = not box_blink_state
        box_blink_frame = 0
    clock.tick(30)  # 30 fps clock
