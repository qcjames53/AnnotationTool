import random
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

#Initialize Pygame Display Window
pygame.init()
display_size = (720,480)
screen = pygame.display.set_mode(display_size, DOUBLEBUF|OPENGL) #Double buffer for monitor refresh rate & OpenGL support in Pygame
gluPerspective(45, (display_size[0]/display_size[1]), 0.1, 50.0) #(FOV, Aspect Ratio, Near Clipping Plane, Far Clipping Plane)
glTranslatef(0.0,0.0, -5) #move camera back 5 units
pygame.display.set_caption("Open GL Test")
clock = pygame.time.Clock()
boxes = [None] #contains None to start at index 1

#Bounding Box Object Class
class BoundingBox():
    """
    Documentation is TBD
    """
    def __init__(self, index, position, rotation, length, width, height, object_type, color_value):
        self.index = index
        self.pos = position
        self.length = length
        self.width = width
        self.height = height
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



vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
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


def draw_bounding_box(index):
    glBegin(GL_LINES)
    glColor3fv(boxes[index].color_value)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(boxes[index].vertices[vertex])
    glEnd()


def draw_cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv((1,0,0))
            glVertex3fv(vertices[vertex])
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


def draw_plane(lines, distance_between_lines):
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


def instantiate_box(position=(0,0,0), rotation=0, width=1.5, height=1, length=2, object_type="Car"):
    color_value = (random.random(), random.random(), random.random())
    box = BoundingBox(position=position, rotation=rotation, length=length, width=width, height=height, object_type=object_type, color_value=color_value, index=len(boxes))
    boxes.append(box)


instantiate_box()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(0.2, 0.1, 1, 0) #rotation (angle, x, y, z)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_plane(10, 0.5)
    draw_bounding_box(1)
    draw_axis()

    pygame.display.flip()
    clock.tick(30) #30 fps clock