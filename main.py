import pygame
import numpy as np


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
    point_as_matrix = np.array([point_3d[0], point_3d[1], point_3d[2], 1])
    point = CAMERA_MATRIX.dot(point_as_matrix)
    point_scaled = np.array([point[0]/point[2],point[1]/point[2],1,1])
    return (point_scaled[0],point_scaled[1])


def render_screen():
    render.fill([0,0,0])
    draw_grid(10,1)


CAMERA_MATRIX = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1/3,-2/3],
                          [0,0,-1/3,1]])
GRID_COLOR = (0, 0, 255)
LINE_WIDTH = 3
RENDER_SIZE = (720, 480)

pygame.init()
render = pygame.display.set_mode(RENDER_SIZE)
pygame.display.set_caption("2D graphics test")
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    # Input handler
    render_screen()
    pygame.display.flip()
    clock.tick(30)