import random
import numpy
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2

#Constants
camera_rot = (30, 0)
box_translation_amount = 1
box_rotation_amount = 1
box_mod_dimension_amount = 1
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

#Global Variables for Pygame & Data Storage
boxes = []
selected_box = 0;
show_ground_plane_grid = False;

#Initialize Pygame Display Window & OpenGL
pygame.init()
render_size = (720, 480)
render = pygame.display.set_mode(render_size, DOUBLEBUF | OPENGL) #Double buffer for monitor refresh rate & OpenGL support in Pygame
gluPerspective(45, (render_size[0] / render_size[1]), 0.1, 50.0) #(FOV, Aspect Ratio, Near Clipping Plane, Far Clipping Plane)
glTranslatef(0.0,0.0, -5) #move camera back 5 units
pygame.display.set_caption("Open GL Test")
clock = pygame.time.Clock()

#Bounding Box Object Class
class BoundingBox():
    """
    Documentation is TBD
    """
    def __init__(self, index, object_type, color_value, position=[0,0,0], rotation=0, width=1.5, height=1, length=2):
        self.index = index
        self.pos = position
        self.pos_init = position
        self.length = length
        self.length_init = length
        self.width = width
        self.width_init = width
        self.height = height
        self.height_init = height
        self.rotation = rotation
        self.rotation_init = rotation
        self.object_type = object_type
        self.color_value = color_value
        self.vertices = None
        self.build_vertices()

    def build_vertices(self):
        self.vertices = (
            (self.pos[0] + self.width / 2, self.pos[1] - self.height / 2, self.pos[2] - self.length / 2),
            (self.pos[0] + self.width / 2, self.pos[1] + self.height / 2, self.pos[2] - self.length / 2),
            (self.pos[0] - self.width / 2, self.pos[1] + self.height / 2, self.pos[2] - self.length / 2),
            (self.pos[0] - self.width / 2, self.pos[1] - self.height / 2, self.pos[2] - self.length / 2),
            (self.pos[0] + self.width / 2, self.pos[1] - self.height / 2, self.pos[2] + self.length / 2),
            (self.pos[0] + self.width / 2, self.pos[1] + self.height / 2, self.pos[2] + self.length / 2),
            (self.pos[0] - self.width / 2, self.pos[1] - self.height / 2, self.pos[2] + self.length / 2),
            (self.pos[0] - self.width / 2, self.pos[1] + self.height / 2, self.pos[2] + self.length / 2)
        )

    def reset(self):
        self.pos = self.pos_init
        self.length = self.length_init
        self.width = self.width_init
        self.height = self.height_init
        self.rotation = self.rotation_init
        self.build_vertices()

    def set_pos(self,pos=(0, 0, 0)):
        self.pos[0] = pos[0]
        self.pos[1] = pos[1]
        self.pos[2] = pos[2]
        self.build_vertices()

    def mod_pos(self,distances=(0, 0, 0)):
        self.pos[0] += distances[0]
        self.pos[1] += distances[1]
        self.pos[2] += distances[2]
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
        self.rotation = rotation
        self.build_vertices()

    def mod_rot(self,rotation):
        self.set_rot(self.rotation + rotation)



def draw_bounding_box(index, selected=False):
    glBegin(GL_LINES)
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
    max = distance / 2
    min = -max
    glBegin(GL_LINES)
    glColor3fv((0,0,1))
    for i in range(0,lines+1):
        glVertex3fv((min + i*distance_between_lines,0,min))
        glVertex3fv((min + i*distance_between_lines,0,max))
        glVertex3fv((min, 0, min + i*distance_between_lines))
        glVertex3fv((max, 0, min + i*distance_between_lines))
    glEnd()


def draw_background_image():
    None


def instantiate_box(position=[0,0,0], rotation=0, width=1.5, height=1, length=2, object_type="Car"):
    color_value = (random.random(), random.random(), random.random())
    box = BoundingBox(position=position, rotation=rotation, length=length, width=width, height=height, object_type=object_type, color_value=color_value, index=len(boxes))
    boxes.append(box)


#load_texture()
glRotatef(camera_rot[0], 1, 0, 0)  #rotation (angle, x, y, z)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN: #Create New Box
                instantiate_box()
            if event.key == pygame.K_PAGEDOWN: #Select Previous Box
                selected_box -= 1
            if event.key == pygame.K_PAGEUP: #Select Next Box
                selected_box += 1
            if pygame.key.get_mods() & pygame.KMOD_ALT: #Toggle Ground Plane Grid Visibility
                show_ground_plane_grid = not show_ground_plane_grid
            if event.key == pygame.K_SPACE: #Reset Selected Box
                boxes[selected_box].reset()
            if event.key == pygame.K_UP: #Translate Selected Box
                boxes[selected_box].mod_pos((0, 0, -box_translation_amount))
            if event.key == pygame.K_DOWN:
                boxes[selected_box].mod_pos((0, 0, box_translation_amount))
            if event.key == pygame.K_LEFT:
                boxes[selected_box].mod_pos((-box_translation_amount, 0, 0))
            if event.key == pygame.K_RIGHT:
                boxes[selected_box].mod_pos((box_translation_amount, 0, 0))
            if event.key == pygame.K_w: #Adjust Height of Selected Box
                boxes[selected_box].mod_height(box_mod_dimension_amount)
            if event.key == pygame.K_s:
                boxes[selected_box].mod_height(-box_mod_dimension_amount)
            if event.key == pygame.K_a: #Adjust Width of Selected Box
                boxes[selected_box].mod_width(box_mod_dimension_amount)
            if event.key == pygame.K_d:
                boxes[selected_box].mod_width(-box_mod_dimension_amount)
            if event.key == pygame.K_q: #Adjust Length / Depth of Selected Box
                boxes[selected_box].mod_length(box_mod_dimension_amount)
            if event.key == pygame.K_e:
                boxes[selected_box].mod_length(-box_mod_dimension_amount)
            if event.key == pygame.K_r: #Adjust Rotation of Selected Box
                boxes[selected_box].mod_rot(box_mod_dimension_amount)
            if event.key == pygame.K_f:
                boxes[selected_box].mod_rot(-box_mod_dimension_amount)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #draw_background_image()
    if (show_ground_plane_grid):
        draw_ground_plane_grid(100, 0.5)
        draw_axis()

    for box in boxes:
        if selected_box == box.index:
            draw_bounding_box(box.index, True)
        else:
            draw_bounding_box(box.index)

    pygame.display.flip()
    clock.tick(30) #30 fps clock