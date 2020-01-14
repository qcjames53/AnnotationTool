# This document contains sample code from https://rdmilligan.wordpress.com/2016/08/27/opengl-shaders-using-python/
#       It was used to handle most of the image display functionality
# The GLUT text modules don't work properly with our setup, so installation of several files from here may be needed:
#      https://python-catalin.blogspot.com/2018/08/pyopengl-fix-attempt-to-call-undefined.html

import math
import numpy
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *

from camera import *


def get_2D_point(point_3d, width, height):
    point_vector = numpy.array([point_3d[0], point_3d[1], point_3d[2], 1])
    print(numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T)
    point = numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T.dot(point_vector)
    # point[0:2] /= point[2]
    return point


def draw(boxes, box_types, frame_number, number_of_frames):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_background_image()
    if in_camera_mode:
        draw_ground_plane_grid(200, 1)
        draw_axis()
    else:
        index = 0
        for box in boxes:
            if selected_box == index and box_blink_state:
                draw_bounding_box(box, index, True)
            else:
                draw_bounding_box(box, index)
            index += 1
    if show_instructions:
        draw_2d_text((2, RENDER_SIZE[1] - 11), "Frame " + str(frame_number) + " of " + str(number_of_frames) +
                     "  ChangeFrame[-,+,0]")
        if in_frame_mode:
            draw_2d_text_center_red("Enter frame: " + str(frame_select_number))
        elif in_camera_mode:
            draw_2d_text((2, 29), "[CAMERA ADJUST MODE]  POS:(" + str(round_output(camera.pos[0])) + "," + str(
                round_output(camera.pos[1])) + "," + str(round_output(camera.pos[2])) + ")   ROT:(" + str(
                round_output(camera.rot[0])) + "," + str(round_output(camera.rot[1])) + ")   FOV:" + str(
                round_output(camera.fov)))
            draw_2d_text((2, 15), "Mode[ALT]  Hide[SHIFT]  Reset[SPACE]")
            draw_2d_text((2, 2),  "Translate[W,A,S,D,E,Q]  Rotate[ARROWS]   FOV[R,F]  PrintPos[P]")
        elif in_place_mode:
            temp_string = "Cancel[DEL/BACKSP]"
            for i in range(0, len(box_types)):
                temp_string += " " + box_types[i][0] + "[" + str(i + 1) + "] "
            draw_2d_text((RENDER_SIZE[0] / 2 - (len(temp_string) * 4), RENDER_SIZE[1] / 2 - 4), temp_string,
                         bg_color=(0.8, 0, 0))
        elif len(boxes) == 0:
            draw_2d_text((2, 29), "[BOX MODE]")
            draw_2d_text((2, 15), "Mode[ALT]  Hide[SHIFT]  New[ENTER]")
        else:
            draw_2d_text((2, 29), "[BOX MODE]  " + boxes[selected_box].to_string_rounded())
            draw_2d_text((2, 15),
                         "Mode[ALT]  Hide[SHIFT]  New[ENTER]  Select[Z,X]  Delete[DEL/BACKSP]              ")
            draw_2d_text((2, 2),
                         "Translate[ARROWS]       Resize[W,A,S,D,Q,E]      Rotate[R,F]         PrintPos[P] ")


def draw_2d_box(pos, size, bg_color):
    new_pos = (pos[0], RENDER_SIZE[1] - pos[1])
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, RENDER_SIZE[0], RENDER_SIZE[1], 0, 0, 1)
    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(bg_color[0], bg_color[1], bg_color[2])
    glVertex2f(new_pos[0], new_pos[1])
    glVertex2f(new_pos[0], new_pos[1] - size[1])
    glVertex2f(new_pos[0] + size[0], new_pos[1])
    glVertex2f(new_pos[0] + size[0], new_pos[1] - size[1])
    glEnd()
    glPopMatrix()


def draw_2d_text(pos, text, color=(1, 1, 1), bg_color=(0, 0, 0)):
    draw_2d_box((pos[0] - TEXT_BORDER, pos[1] - 2 * TEXT_BORDER), (len(text) * 8 + 2 * TEXT_BORDER, 13 + TEXT_BORDER), bg_color)
    blending = False
    if glIsEnabled(GL_BLEND) :
        blending = True
    # glEnable(GL_BLEND)
    glColor3f(color[0], color[1], color[2])
    glWindowPos2f(pos[0],pos[1])
    for ch in text:
        glutBitmapCharacter(OpenGL.GLUT.GLUT_BITMAP_8_BY_13, ctypes.c_int(ord(ch)))
    if not blending :
        glDisable(GL_BLEND)


def draw_2d_text_center_red(string):
    draw_2d_text((RENDER_SIZE[0] / 2 - (len(string) * 4), RENDER_SIZE[1] / 2 - 4), string, bg_color=(0.8, 0, 0))


def draw_3d_text(pos, text, color=(1, 1, 1)):
            blending = False
            if glIsEnabled(GL_BLEND):
                blending = True
            # glEnable(GL_BLEND)
            glColor3f(color[0], color[1], color[2])
            glRasterPos3f(pos[0],pos[1],pos[2])
            for ch in text:
                glutBitmapCharacter(OpenGL.GLUT.GLUT_BITMAP_8_BY_13, ctypes.c_int(ord(ch)))
            if not blending:
                glDisable(GL_BLEND)


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


def draw_background_image():
    # create projection matrix
    fov = math.radians(45.0)
    f = 1.0 / math.tan(fov / 2.0)
    zNear = 0.1
    zFar = 100.0
    aspect = RENDER_SIZE[0] / RENDER_SIZE[1]
    pMatrix = numpy.array([
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (zFar + zNear) / (zNear - zFar), -1.0,
        0.0, 0.0, 2.0 * zFar * zNear / (zNear - zFar), 0.0], numpy.float32)
    # create modelview matrix
    mvMatrix = numpy.array([
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, -3.6, 1.0], numpy.float32)
    # use shader program
    glUseProgram(shader_program)
    # set uniforms
    glUniformMatrix4fv(uPMatrix, 1, GL_FALSE, pMatrix)
    glUniformMatrix4fv(uMVMatrix, 1, GL_FALSE, mvMatrix)
    glUniform1i(uBackgroundTexture, 0)
    # enable attribute arrays
    glEnableVertexAttribArray(aVert)
    glEnableVertexAttribArray(aUV)
    # set vertex and UV buffers
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
    glVertexAttribPointer(aVert, 3, GL_FLOAT, GL_FALSE, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer)
    glVertexAttribPointer(aUV, 2, GL_FLOAT, GL_FALSE, 0, None)
    # bind background texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, backgroundTexture)
    # draw
    glDrawArrays(GL_TRIANGLES, 0, 6)
    # disable attribute arrays
    glDisableVertexAttribArray(aVert)
    glDisableVertexAttribArray(aUV)
    glUseProgram(0)


def draw_bounding_box(box, index, selected=False):
    if box.truncated == 0:
        glBegin(GL_LINES)
        if selected:
            glColor3fv((1, 1, 1))
        else:
            glColor3fv(box.color_value)
        for edge in BOX_EDGE_RENDER_ORDER:
            for vertex in edge:
                glVertex3fv(box.vertices[vertex])
        glEnd()
        draw_3d_text(box.location, box.type + str(index))


def draw_ground_plane_grid(lines, distance_between_lines):
    distance = lines * distance_between_lines
    plane_max = distance / 2
    plane_min = -plane_max
    glBegin(GL_LINES)
    glColor3fv((0.35,0.5,0.25))
    for i in range(0,lines+1):
        glVertex3fv((plane_min + i*distance_between_lines,0,plane_min))
        glVertex3fv((plane_min + i*distance_between_lines,0,plane_max))
        glVertex3fv((plane_min, 0, plane_min + i*distance_between_lines))
        glVertex3fv((plane_max, 0, plane_min + i*distance_between_lines))
    glEnd()


def input_handler(event, boxes, box_types, click_box=-1):
    global box_blink_frame
    global box_blink_state
    global frame_select_number
    global in_camera_mode
    global in_frame_mode
    global in_place_mode
    global selected_box
    global show_instructions

    if click_box != -1:
        selected_box = click_box
        box_blink_frame = 0
        box_blink_state = True
        return ""

    ##### GLOBAL KEYS (Never Disabled) #####
    # File output of data
    if event.key == pygame.K_o:
        draw_2d_text_center_red("Saving To File")
        return "output"
    # Next Image / Previous Image controls
    elif event.key == pygame.K_EQUALS:
        draw_2d_text_center_red("Compiling new image shader")
        return "f+"
    elif event.key == pygame.K_MINUS:
        draw_2d_text_center_red("Compiling new image shader")
        return "f-"
    elif event.key == pygame.K_0:
        in_frame_mode = True
        show_instructions = True
    # Toggle mode
    elif pygame.key.get_mods() & pygame.KMOD_ALT:
        in_camera_mode = not in_camera_mode
    # Toggle instruction visibility
    elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
        show_instructions = not show_instructions

    ##### FRAME SELECT MODE #####
    if in_frame_mode:
        if event.key == pygame.K_RETURN:
            in_frame_mode = False
            draw_2d_text_center_red("Compiling new image shader")
            temp = frame_select_number
            frame_select_number = 0
            return "f#" + str(temp)
        elif event.key == pygame.K_BACKSPACE:
            frame_select_number = math.floor(frame_select_number / 10)
        elif pygame.key.name(event.key).isdigit():
            frame_select_number = frame_select_number * 10 + int(pygame.key.name(event.key))

    ##### CAMERA ADJUSTMENT MODE #####
    elif in_camera_mode:
        # Camera reset
        if event.key == pygame.K_SPACE:
            camera.reset()
            set_camera()
        # Camera translation
        elif event.key == pygame.K_w:
            camera.move(
                z=CAMERA_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_s:
            camera.move(
                z=-CAMERA_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_a:
            camera.move(
                x=-CAMERA_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_d:
            camera.move(
                x=CAMERA_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_q:
            camera.move(
                y=CAMERA_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_e:
            camera.move(
                y=-CAMERA_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        # Camera FOV manip
        elif event.key == pygame.K_r:
            camera.move(
                fov=CAMERA_MOD_FOV_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_f:
            camera.move(
                fov=-CAMERA_MOD_FOV_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        # Camera  rotation
        elif event.key == pygame.K_UP:
            camera.move(
                rot_a=CAMERA_ROTATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_DOWN:
            camera.move(
                rot_a=-CAMERA_ROTATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_LEFT:
            camera.move(
                rot_b=CAMERA_ROTATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        elif event.key == pygame.K_RIGHT:
            camera.move(
                rot_b=-CAMERA_ROTATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
            set_camera()
        # Print positioning
        elif event.key == pygame.K_p:
            print("Camera position: " + str(camera.pos))
            print("Camera rotation: " + str(camera.rot))
            print("Camera FOV: " + str(camera.fov))

    ##### BOX INSTANTIATION MODE #####
    elif in_place_mode:
        if pygame.key.name(event.key).isdigit() and 0 < int(pygame.key.name(event.key)) <= len(box_types):
            box_blink_frame = 0
            box_blink_state = True
            in_place_mode = False
            selected_box = len(boxes)
            return "b#" + pygame.key.name(event.key)
        elif (event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE):
            in_place_mode = False

    ##### BOX ADJUSTMENT MODE #####
    else:
        if event.key == pygame.K_RETURN:
            in_place_mode = True
            show_instructions = True
        elif len(boxes) == 0:  # Program will crash if any below values called with 0 index and no boxes
            None
        elif (event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE):
            boxes.pop(selected_box)
            selected_box = min(selected_box, len(boxes) - 1)  # clamp selected_box to usable range
            box_blink_frame = 0
            box_blink_state = True
        # Box selection
        elif (event.key == pygame.K_PAGEDOWN or event.key == pygame.K_z) and selected_box > 0:  # Select Previous Box
            selected_box -= 1
            box_blink_frame = 0
            box_blink_state = True
        elif (event.key == pygame.K_PAGEUP or event.key == pygame.K_x) and selected_box < len(
                boxes) - 1:  # Select Next Box
            selected_box += 1
            box_blink_frame = 0
            box_blink_state = True
        # Adjust selected box translation
        elif event.key == pygame.K_UP:
            boxes[selected_box].mod_location(
                (0, 0, BOX_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
        elif event.key == pygame.K_DOWN:
            boxes[selected_box].mod_location(
                (0, 0, -BOX_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
        elif event.key == pygame.K_LEFT:
            boxes[selected_box].mod_location(
                (-BOX_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
        elif event.key == pygame.K_RIGHT:
            boxes[selected_box].mod_location(
                (BOX_TRANSLATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
        # Adjust selected box dimensions
        elif event.key == pygame.K_w:
            boxes[selected_box].mod_dimensions(
                (BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
        elif event.key == pygame.K_s and boxes[selected_box].dimensions[0] - (
                BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:
            boxes[selected_box].mod_dimensions(
                (-BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
        elif event.key == pygame.K_a and boxes[selected_box].dimensions[1] - (BOX_MOD_DIMENSION_AMOUNT * (
        CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:  # Adjust Width of Selected Box
            boxes[selected_box].mod_dimensions(
                (0, -BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0))
        elif event.key == pygame.K_d:
            boxes[selected_box].mod_dimensions(
                (0, BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0))
        elif event.key == pygame.K_q and boxes[selected_box].dimensions[2] - (BOX_MOD_DIMENSION_AMOUNT * (
        CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:  # Adjust Length / Depth of Selected Box
            boxes[selected_box].mod_dimensions(
                (0, 0, -BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
        elif event.key == pygame.K_e:
            boxes[selected_box].mod_dimensions(
                (0, 0, BOX_MOD_DIMENSION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
        # Adjust selected box rotation
        elif event.key == pygame.K_r:
            boxes[selected_box].mod_rotation_y(
                BOX_ROTATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
        elif event.key == pygame.K_f:
            boxes[selected_box].mod_rotation_y(
                -BOX_ROTATION_AMOUNT * (CTRL_MULTI if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
        # Print selected box
        elif event.key == pygame.K_p:
            #print(boxes[selected_box].to_string())
            print("2D point of center: " + str(get_2D_point(boxes[selected_box].location, RENDER_SIZE[0], RENDER_SIZE[1])))

    # Catch-all return for function
    return ""


def round_output(x):
    return "%.3f" % round(x,3)


def set_background_image(backgroundImage):
    # set background texture
    global backgroundTexture
    backgroundImageData = numpy.array(list(backgroundImage.getdata()), numpy.uint8)
    backgroundTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, backgroundTexture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, backgroundImage.size[0], backgroundImage.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE,
                 backgroundImageData)


def set_camera():
    numpy.set_printoptions(suppress=True)

    glLoadIdentity()
    print(numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T)
    gluPerspective(camera.fov, (RENDER_SIZE[0] / RENDER_SIZE[1]), camera.ncp,
                   camera.fcp)  # (FOV, Aspect Ratio, Near Clipping Plane, Far Clipping Plane)
    print(numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T)
    glTranslatef(camera.pos[0], camera.pos[1], camera.pos[2])  # move camera
    print(numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX)).T)
    glRotatef(camera.rot[0], 1, 0, 0)  # rotation of camera (angle, x, y, z)
    glRotatef(camera.rot[1], 0, 1, 0)


def render_screen(boxes, box_types, frame, number_of_frames):
    global box_blink_frame
    global box_blink_state

    output = ""

    # Handle input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            output = input_handler(event, boxes, box_types)

        if pygame.mouse.get_pressed()[0] == 1 and len(boxes) > 0:
            mouse_pos = pygame.mouse.get_pos()
            mouse_x = mouse_pos[0]
            mouse_y = mouse_pos[1]
            for index, box in enumerate(boxes):
                loc_x = box.location[0]
                loc_z = box.location[2]
                pixel_x = 360 + ((loc_x/0.1)*5)
                pixel_y = 240 - ((loc_z / 0.1) * 2)
                if pixel_x + 50 >= mouse_x and pixel_x - 50 <= mouse_x and pixel_y + 100 >= mouse_y and pixel_y - 100 <= mouse_y:
                    output = input_handler(event, boxes, box_types, index)

    # Increment box blinks
    box_blink_frame += 1
    if box_blink_frame >= BOX_BLINK_SPEED:
        box_blink_state = not box_blink_state
        box_blink_frame = 0

    # Update alpha of selected box
    if len(boxes) > selected_box >= 0:
        boxes[selected_box].build_alpha(camera.get_pos_copy())

    # Draw screen at 30 fps
    if output == "":
        draw(boxes, box_types, frame, number_of_frames)
    pygame.display.flip()
    clock.tick(30)
    return output


# Constants <- these can be tweaked
BOX_BLINK_SPEED = 15
BOX_EDGE_RENDER_ORDER = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7), (8, 9))
BOX_MOD_DIMENSION_AMOUNT = 0.1
BOX_ROTATION_AMOUNT = math.radians(1)
BOX_TRANSLATION_AMOUNT = 0.1
CAMERA_MOD_FOV_AMOUNT = 0.1
CAMERA_ROTATION_AMOUNT = 0.1
CAMERA_TRANSLATION_AMOUNT = 0.05
CTRL_MULTI = 10
RENDER_SIZE = (720, 480)
TEXT_BORDER = 2

# Global Variables for PyGame & Data Storage
box_blink_frame = 0
box_blink_state = False
camera = Camera([0.0, -0.55, -43.55], [192.5, 0.6], 16.6, 0.1, 200)  # Camera object to store camera variables
frame_select_number = 0
in_camera_mode = False  # True if user is adjusting camera
in_frame_mode = False  # True if user is selecting a frame
in_place_mode = False  # True if user is placing box
selected_box = 0  # Index of selected box in boxes array
show_instructions = True  # True if instructions are visible of screen

# Initialize PyGame Display Window & OpenGL
pygame.init()
glutInit()
render = pygame.display.set_mode(RENDER_SIZE, DOUBLEBUF | OPENGL)  # Double buffer for monitor refresh rate & OpenGL support in Pygame
set_camera()
pygame.display.set_caption("Open GL Test")
clock = pygame.time.Clock()

# OpenGL Image display shader setup
vertexShader = """
    #version 330 core

    attribute vec3 vert;
    attribute vec2 uV;
    uniform mat4 mvMatrix;
    uniform mat4 pMatrix;
    out vec2 UV;

    void main() {
      gl_Position = pMatrix * mvMatrix * vec4(vert, 1.0);
      UV = uV;
    }
"""
fragmentShader = """
    #version 330 core

    in vec2 UV;
    uniform sampler2D backgroundTexture;
    out vec3 colour;

    void main() {
      colour = texture(backgroundTexture, UV).rgb;
    }
"""
vs = compileShader(vertexShader, GL_VERTEX_SHADER)
fs = compileShader(fragmentShader, GL_FRAGMENT_SHADER)
shader_program = compileProgram(vs, fs)

# obtain uniforms and attributes
aVert = glGetAttribLocation(shader_program, "vert")
aUV = glGetAttribLocation(shader_program, "uV")
uPMatrix = glGetUniformLocation(shader_program, 'pMatrix')
uMVMatrix = glGetUniformLocation(shader_program, "mvMatrix")
uBackgroundTexture = glGetUniformLocation(shader_program, "backgroundTexture")

# set background vertices
backgroundVertices = [
    -2.0, 1.5, 0.0,
    -2.0, -1.5, 0.0,
    2.0, 1.5, 0.0,
    2.0, 1.5, 0.0,
    -2.0, -1.5, 0.0,
    2.0, -1.5, 0.0]

vertexBuffer = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
vertexData = numpy.array(backgroundVertices, numpy.float32)
glBufferData(GL_ARRAY_BUFFER, 4 * len(vertexData), vertexData, GL_STATIC_DRAW)

# set background UV
backgroundUV = [
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    1.0, 1.0]

uvBuffer = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, uvBuffer)
uvData = numpy.array(backgroundUV, numpy.float32)
glBufferData(GL_ARRAY_BUFFER, 4 * len(uvData), uvData, GL_STATIC_DRAW)
backgroundTexture = glGenTextures(1)
