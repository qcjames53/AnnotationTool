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
# gluPerspective(45, (render_size[0] / render_size[1]), 0.1, 50.0)
clock = pygame.time.Clock()

near_clipping_plane = 0.1
far_clipping_plane = 50
persp_temp_a = near_clipping_plane + far_clipping_plane
persp_temp_b = near_clipping_plane * far_clipping_plane
persp = np.array([[583.2829786373293,                 0,       -320.0,            0],
                  [                0, 579.4112549695428,       -240.0,            0],
                  [                0,                 0, persp_temp_a, persp_temp_b],
                  [                0,                 0,           -1,            0]])
glLoadIdentity()
# glOrtho(0, render_size[0], render_size[1], 0, near_clipping_plane, far_clipping_plane);
glOrtho(-1 * render_size[0]/2, render_size[0]/2, -1 * render_size[1]/2, render_size[1]/2, near_clipping_plane, far_clipping_plane);
glMultMatrixd(persp)


glTranslatef(0.0,0.0, -5)

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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glRotatef(1, 0, 1, 0.2)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    draw_ground_plane_grid(100, 0.5)

    glBegin(GL_LINES)
    glColor3fv((1, 1, 1))
    for edge in box_edge_render_order:
        for vertex in edge:
            glVertex3fv(box_vertices[vertex])
    glEnd()

    pygame.display.flip()
    clock.tick(30)