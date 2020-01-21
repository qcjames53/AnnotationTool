import numpy as np
import pygame
import sys

def draw_axis():
    draw_line((0, 0, 0), (1, 0, 0), (255, 0, 0))
    draw_line((0, 0, 0), (0, 1, 0), (0, 255, 0))
    draw_line((0, 0, 0), (0, 0, 1), (0, 0, 255))

def draw_grid(line_count, distance_between):
    distance = line_count * distance_between
    plane_max = distance/2
    plane_min = -plane_max
    for i in range(0, line_count + 1):
        coord = plane_min + i * distance_between
        draw_line((coord,0,plane_min),(coord,0,plane_max),GRID_COLOR)
        draw_line((plane_min,0,coord),(plane_max,0,coord),GRID_COLOR)


def draw_line(c1, c2, color):
    draw_line_screen(get_screen_point(c1),get_screen_point(c2),color)


def draw_line_screen(px1, px2, color):
    pygame.draw.line(render, color, px1, px2, LINE_WIDTH)


def get_screen_point(point_3d):
    point_as_matrix = np.array([point_3d[0] + ORIGIN[0], point_3d[1] + ORIGIN[1], point_3d[2] + ORIGIN[2], 1])
    point = CAMERA_MATRIX.dot(point_as_matrix)
    point_scaled = np.array([point[0]/point[2],point[1]/point[2],1,1])
    return (point_scaled[0],point_scaled[1])


def render_screen():
    render.fill([0,0,0])
    draw_grid(10,1)
    draw_axis()


CAMERA_MATRIX = np.array([[721.5377    ,   0.        , 609.5593    ,  44.85728   ],
                          [  0.        , 721.5377    , 172.854     ,   0.2163791 ],
                          [  0.        ,   0.        ,   1.        ,   0.00274588],
                          [  0.        ,   0.        ,   0.        ,   1.        ]])
GRID_COLOR = (80, 80, 80)
LINE_WIDTH = 1
ORIGIN = (0,-3,-13)
RENDER_SIZE = (720, 480)

pygame.init()
render = pygame.display.set_mode(RENDER_SIZE)
pygame.display.set_caption("2D graphics test")
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Input handler
    render_screen()
    pygame.display.flip()
    clock.tick(30)