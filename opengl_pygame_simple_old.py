import pygame
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

box_edge_render_order = (
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

box_vertices = (
    (1,-1,-1),
    (1,1,-1),
    (-1,1,-1),
    (-1,-1,-1),
    (1,-1,1),
    (1,1,1),
    (-1,-1,1),
    (-1,1,1)
)

pygame.init()
render_size = (720, 480)
render = pygame.display.set_mode(render_size, DOUBLEBUF | OPENGL)
gluPerspective(45, (render_size[0] / render_size[1]), 0.1, 50.0)
clock = pygame.time.Clock()
glTranslatef(0.0,0.0, -5)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(1, 0, 1, 0.2)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glBegin(GL_LINES)
    glColor3fv((1, 1, 1))
    for edge in box_edge_render_order:
        for vertex in edge:
            glVertex3fv(box_vertices[vertex])
    glEnd()

    pygame.display.flip()
    clock.tick(30)