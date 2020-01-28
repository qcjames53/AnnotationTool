import numpy as np
import pygame
import sys

from bounding_box import *
from button import *

pygame.init()


def build_controls():
    global buttons
    buttons = []
    current_x = 4
    current_y = RENDER_SIZE[1] - 72
    r_text = [["Select Frame","+","-","|","New Car","New Pedestrian","New Cyclist","|","Select Box","+","-"],
              ["x+","x-","z+","z-","r+","r-","l+","l-","w+","w-","h+","h-"]]
    for row in r_text:
        for text in row:
            if text == "|":
                current_x += 12
            else:
                temp = Button((current_x, current_y), text)
                current_x += temp.get_len()
                buttons.append(temp)
        current_x = 4
        current_y += 24
    global message_text
    message_text = "Setup complete"


def button_handler(index):
    global background_image_index
    global message_text
    global input_state
    if index == 0:  # Input Frame Number
        input_state = 2
    elif index == 1:  # + Frame
        background_image_index += 1
        if background_image_index > background_image_index_max:
            background_image_index = background_image_index_max
        set_background_image()
        message_text = "Set frame to " + str(background_image_index)
    elif index == 2:  # - Frame
        background_image_index -= 1
        if background_image_index < background_image_index_min:
            background_image_index = background_image_index_min
        set_background_image()
        message_text = "Set frame to " + str(background_image_index)



def draw_axis():
    draw_line((0, 0, 0), (1, 0, 0), (255, 0, 0))
    draw_line((0, 0, 0), (0, 1, 0), (0, 255, 0))
    draw_line((0, 0, 0), (0, 0, 1), (0, 0, 255))


def draw_bounding_boxes(IgnoreCulling = False):
    for i in range(0,len(boxes)):
        if IgnoreCulling or boxes[i].truncated == 0:
            color = boxes[i].color_value
            if i == selected_box and blink_animation_frame > blink_animation_time:
                color = C_WHITE
            for edge in BOX_EDGE_RENDER_ORDER:
                draw_line(boxes[i].vertices[edge[0]],boxes[i].vertices[edge[1]],color)


def draw_controls():
    control_origin = (0,IMAGE_SIZE[1])
    control_size = (RENDER_SIZE[0], 102)
    pygame.draw.rect(render, C_BACKGROUND, (control_origin,control_size))
    pygame.draw.line(render, C_HIGHLIGHT, (control_origin[0], control_origin[1]+2), (control_origin[0]+control_size[0],control_origin[1]+2),2)

    # Row 1 (Message Center)
    window_x = control_origin[0] + 4
    window_y = control_origin[1] + 6
    window_width = (control_size[0] - 4) - window_x
    window_height = 24
    pygame.draw.rect(render, C_SHADOW, ((window_x,window_y),(window_width, window_height)))
    pygame.draw.rect(render, C_WHITE, ((window_x+2,window_y+2),(window_width-4, window_height-4)))
    pygame.draw.line(render, C_HIGHLIGHT, (window_x+2, window_y + window_height-4), (window_width+1, window_y + window_height-4), 2)
    message = CONTROL_FONT.render(message_text, True, C_BLACK)
    render.blit(message, (window_x+4, window_y+2))

    # Rows 2 - 4
    for index in range(0,len(buttons)):
        draw_control_button(index)

def draw_control_button(index):
    pos = buttons[index].get_ul()
    len = buttons[index].get_len()
    text = buttons[index].get_text()
    pygame.draw.rect(render, C_BLACK, (pos, (len, 24)))
    if buttons[index].get_triggered():
        pygame.draw.rect(render, C_HIGHLIGHT, ((pos[0], pos[1]), (len - 1, 23)))
        pygame.draw.rect(render, C_BACKGROUND, ((pos[0] + 1, pos[1] + 1), (len - 3, 20)))
        pygame.draw.rect(render, C_SHADOW, ((pos[0] + 2, pos[1] + 2), (len - 4, 19)))
        if mouse_pressed == False:
            buttons[index].remove_trigger()
    else:
        pygame.draw.rect(render, C_SHADOW, ((pos[0],pos[1]), (len - 1, 23)))
        pygame.draw.rect(render, C_HIGHLIGHT, ((pos[0] + 1, pos[1] + 1), (len - 3, 20)))
        pygame.draw.rect(render, C_BACKGROUND, ((pos[0] + 2, pos[1] + 2), (len - 4, 19)))
    message = CONTROL_FONT.render(text, True, C_BLACK)
    render.blit(message, (pos[0] + 6, pos[1] + 2))


# Draws a grid to the boundaries of the culling area every 1 unit
def draw_grid():
    x_min = CULLING_AREA[0][0]
    x_max = CULLING_AREA[1][0]
    y_min = CULLING_AREA[0][2]
    y_max = CULLING_AREA[1][2]
    for i in range(x_min, x_max+1):
        draw_line((i,0,y_min),(i,0,y_max),GRID_COLOR)
    for i in range(y_min, y_max+1):
        draw_line((x_min, 0, i), (x_max, 0, i), GRID_COLOR)


def draw_line(c1, c2, color):
    draw_line_screen(get_screen_point(c1),get_screen_point(c2),color)


def draw_line_screen(px1, px2, color):
    pygame.draw.line(render, color, px1, px2, LINE_WIDTH)


def get_ground_point(point_2d):
    None


def get_screen_point(point_3d):
    point_as_matrix = np.array([point_3d[0] + camera_pos[0], point_3d[1] + camera_pos[1], point_3d[2] + camera_pos[2], 1])
    point = camera_matrix.dot(point_as_matrix)
    point_scaled = np.array([point[0]/point[2],point[1]/point[2],1,1])
    return (point_scaled[0],point_scaled[1])


def input_handler(event):
    global background_image_index
    global message_text
    global input_state

    if input_state == 0 or input_state == 1:
        if event.type == pygame.MOUSEBUTTONDOWN:
            global mouse_pressed
            mouse_pressed = True
            mouse = pygame.mouse.get_pos()
            for i in range(0, len(buttons)):
                if buttons[i].is_triggered(mouse):
                    button_handler(i)
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False

    if input_state == 2:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if background_image_index < background_image_index_min:
                    background_image_index = background_image_index_min
                elif background_image_index > background_image_index_max:
                    background_image_index = background_image_index_max
                input_state = 0
                mouse_pressed = False
                set_background_image()
                message_text = "Set frame to " + str(background_image_index)
            elif event.key == pygame.K_BACKSPACE:
                background_image_index = math.floor(background_image_index / 10)
            elif pygame.key.name(event.key).isdigit():
                background_image_index = background_image_index * 10 + int(pygame.key.name(event.key))


# Adds a bounding box to the current frame based on several parameters
def instantiate_box(index):
    # Constructor: (type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, color_value)
    new_box = BoundingBox(BOX_TYPES[index][0], 0, 0, 0, [0,0,0,0], BOX_TYPES[index][1], (0,0,0), 0, BOX_TYPES[index][2])
    selected_box = len(boxes)
    boxes.append(new_box)


def is_not_culled(point):
    x = CULLING_AREA[0][0] <= point[0] <= CULLING_AREA[1][0]
    y = CULLING_AREA[0][1] <= point[1] <= CULLING_AREA[1][1]
    z = CULLING_AREA[0][2] <= point[2] <= CULLING_AREA[1][2]
    return x and y and z


def render_screen():
    if background_image == None:
        render.fill([0,0,0])
    else:
        render.blit(background_image, ((0,0),(IMAGE_SIZE[0],IMAGE_SIZE[1])))
    draw_grid()
    draw_axis()
    draw_bounding_boxes(IgnoreCulling=True)
    draw_controls()


def set_background_image():
    padding = ""
    while len(padding + str(background_image_index)) < 6:
        padding += "0"
    filename = "image_2/img" + str(padding) + str(background_image_index) + ".jpg"
    global background_image
    background_image = pygame.image.load(filename)


# Constants
C_BACKGROUND = (192, 192, 192)
C_HIGHLIGHT = (230,230,230)
C_SHADOW = (160,160,160)
C_BLACK = (0,0,0)
C_WHITE = (255,255,255)
C_RED = (255,0,0)
C_GREEN = (0, 255, 0)
C_BLUE = (0, 0, 255)

BOX_EDGE_RENDER_ORDER = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7), (8, 9))
BOX_TYPES = (("Car", (1.7, 2.0, 5.0), C_RED),  # (name, dimensions(h,w,l), color_value)
             ("Cyclist", (1.8,0.5,2.0), C_GREEN),
             ("Pedestrian", (1.7,0.5,0.5), C_BLUE))
CONTROL_FONT = pygame.font.Font("resources/ubuntu_mono.ttf", 16)
CULLING_AREA = ((-5,-5,-5),(5,10,5))  # Will only draw boxes inside these two coordinates
GRID_COLOR = (80, 80, 80)
LINE_WIDTH = 1
IMAGE_SIZE = (720, 480)
RENDER_SIZE = (IMAGE_SIZE[0], IMAGE_SIZE[1] + 102)

# Camera Variables
camera_matrix = np.array([[721.5377    , 0.        , IMAGE_SIZE[0] / 2, 44.85728],
                          [0.        , 721.5377    , IMAGE_SIZE[1] / 2, 0.2163791],
                          [  0.        ,   0.        ,   1.            ,   0.00274588],
                          [  0.        ,   0.        ,   0.            ,   1.        ]])
camera_pos = [0, -2, -13]
camera_rot = [0, 0]

# Box Variables
blink_animation_frame = 0
blink_animation_time = 15
boxes = []
selected_box = -1

# Misc Variables
background_image = None
background_image_index = 0
background_image_index_min = 1
background_image_index_max = 1000
buttons = []
input_state = 0  # Input states: 0 - Nothing selected, camera adjustments. 1 - Box selected, box adjustments. 2 - Input for frame no. 3 - Input for box no.
message_text = ""
mouse_pressed = False

render = pygame.display.set_mode(RENDER_SIZE)
pygame.display.set_caption("Bounding Box Visualization")
clock = pygame.time.Clock()
build_controls()

while True:
    # Handle Inputs
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        else:
            input_handler(event)

    # Pre-Render Operations

    # Display input for input state 2
    if input_state == 2:
        if blink_animation_frame < 15:
            message_text = "Input frame number: " + str(background_image_index) + "█"
        else:
            message_text = "Input frame number: " + str(background_image_index)

    # Render Screen
    render_screen()

    # Post-Render Operations
    blink_animation_frame += 1
    if blink_animation_frame > blink_animation_time * 2:
        blink_animation_frame = 0
    pygame.display.flip()
    clock.tick(30)