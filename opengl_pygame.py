# This document contains sample code from https://rdmilligan.wordpress.com/2016/08/27/opengl-shaders-using-python/
# It was used to handle most of the image display functionality

# Classes: car, pedestrian, cyclist

import random
import numpy
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from PIL import Image

# Constants
text_border = 2
show_full_data = False
ctrl_multi = 10
background_image = "test_image.png"
camera_translation_amount = 0.05
camera_rotation_amount = 0.1
camera_mod_fov_amount = 1
box_translation_amount = 0.1
box_rotation_amount = math.radians(1)
box_mod_dimension_amount = 0.1
box_blink_speed = 15
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
    (5,7),
    (8,9)  # forward-facing line
    )
box_types = (("Car", (1,0,0), (2.0,1.7,5.0)), #(name, default_color, default_size(w,h,l))
             ("Cyc", (0,1,0), (0.5,1.8,1.0)),
             ("Ped", (0,0,1), (0.5,1.7,0.5)))

# Global Variables for PyGame & Data Storage
boxes = []
selected_box = 0
in_camera_mode = False
in_place_mode = False
show_instructions = True
box_blink_frame = 0
box_blink_state = False
camera_rot = [30.9, -1.7]
camera_rot_init = camera_rot.copy()
camera_pos = [0.35, -0.55, -17.9]
camera_pos_init = camera_pos.copy()
camera_fov = 43
camera_fov_init = camera_fov
camera_ncp = 0.1
camera_fcp = 100

def set_camera():
    glLoadIdentity()
    gluPerspective(camera_fov, (render_size[0] / render_size[1]), camera_ncp,
                   camera_fcp)  # (FOV, Aspect Ratio, Near Clipping Plane, Far Clipping Plane)
    glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])  # move camera
    glRotatef(camera_rot[0], 1, 0, 0)  # rotation of camera (angle, x, y, z)
    glRotatef(camera_rot[1], 0, 1, 0)

# Initialize PyGame Display Window & OpenGL
pygame.init()
glutInit()
render_size = (720, 480)
render = pygame.display.set_mode(render_size, DOUBLEBUF | OPENGL)  # Double buffer for monitor refresh rate & OpenGL support in Pygame
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

# set background texture
backgroundImage = Image.open(background_image)
backgroundImageData = numpy.array(list(backgroundImage.getdata()), numpy.uint8)
backgroundTexture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, backgroundTexture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, backgroundImage.size[0], backgroundImage.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE,
             backgroundImageData)


# Bounding Box Object Class
class BoundingBox():
    """
    Documentation is TBD
    """
    def __init__(self, object_type, color_value, position=(0,0,0), rotation=0, size=(1,1,1)):
        self.pos = position
        self.pos_init = position
        self.length = size[2]
        self.length_init = self.length
        self.width = size[0]
        self.width_init = self.width
        self.height = size[1]
        self.height_init = self.height
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
        v0_x2 = (self.pos[0] + (((v0_x-self.pos[0]) * math.cos(self.rot)) - ((v0_z-self.pos[2]) * math.sin(self.rot))))
        v0_z2 = (self.pos[2] + (((v0_z-self.pos[2]) * math.cos(self.rot)) + ((v0_x-self.pos[0]) * math.sin(self.rot))))
        v0 = (v0_x2,v0_y,v0_z2)

        v1_x = self.pos[0] + self.width / 2
        v1_y = self.pos[1] + self.height
        v1_z = self.pos[2] - self.length / 2
        v1_x2 = self.pos[0] + ((v1_x-self.pos[0]) * math.cos(self.rot) - (v1_z-self.pos[2]) * math.sin(self.rot))
        v1_z2 = self.pos[2] + ((v1_z-self.pos[2]) * math.cos(self.rot) + (v1_x-self.pos[0]) * math.sin(self.rot))
        v1 = (v1_x2, v1_y, v1_z2)

        v2_x = self.pos[0] - self.width / 2
        v2_y = self.pos[1] + self.height
        v2_z = self.pos[2] - self.length / 2
        v2_x2 = self.pos[0] + ((v2_x-self.pos[0]) * math.cos(self.rot) - (v2_z-self.pos[2]) * math.sin(self.rot))
        v2_z2 = self.pos[2] + ((v2_z-self.pos[2]) * math.cos(self.rot) + (v2_x-self.pos[0]) * math.sin(self.rot))
        v2 = (v2_x2, v2_y, v2_z2)

        v3_x = self.pos[0] - self.width / 2
        v3_y = self.pos[1]
        v3_z = self.pos[2] - self.length / 2
        v3_x2 = self.pos[0] + ((v3_x-self.pos[0]) * math.cos(self.rot) - (v3_z-self.pos[2]) * math.sin(self.rot))
        v3_z2 = self.pos[2] + ((v3_z-self.pos[2]) * math.cos(self.rot) + (v3_x-self.pos[0]) * math.sin(self.rot))
        v3 = (v3_x2, v3_y, v3_z2)

        v4_x = self.pos[0] + self.width / 2
        v4_y = self.pos[1]
        v4_z = self.pos[2] + self.length / 2
        v4_x2 = self.pos[0] + ((v4_x-self.pos[0]) * math.cos(self.rot) - (v4_z-self.pos[2]) * math.sin(self.rot))
        v4_z2 = self.pos[2] + ((v4_z-self.pos[2]) * math.cos(self.rot) + (v4_x-self.pos[0]) * math.sin(self.rot))
        v4 = (v4_x2, v4_y, v4_z2)

        v5_x = self.pos[0] + self.width / 2
        v5_y = self.pos[1] + self.height
        v5_z = self.pos[2] + self.length / 2
        v5_x2 = self.pos[0] + ((v5_x-self.pos[0]) * math.cos(self.rot) - (v5_z-self.pos[2]) * math.sin(self.rot))
        v5_z2 = self.pos[2] + ((v5_z-self.pos[2]) * math.cos(self.rot) + (v5_x-self.pos[0]) * math.sin(self.rot))
        v5 = (v5_x2, v5_y, v5_z2)

        v6_x = self.pos[0] - self.width / 2
        v6_y = self.pos[1]
        v6_z = self.pos[2] + self.length / 2
        v6_x2 = self.pos[0] + ((v6_x-self.pos[0]) * math.cos(self.rot) - (v6_z-self.pos[2]) * math.sin(self.rot))
        v6_z2 = self.pos[2] + ((v6_z-self.pos[2]) * math.cos(self.rot) + (v6_x-self.pos[0]) * math.sin(self.rot))
        v6 = (v6_x2, v6_y, v6_z2)

        v7_x = self.pos[0] - self.width / 2
        v7_y = self.pos[1] + self.height
        v7_z = self.pos[2] + self.length / 2
        v7_x2 = self.pos[0] + ((v7_x-self.pos[0]) * math.cos(self.rot) - (v7_z-self.pos[2]) * math.sin(self.rot))
        v7_z2 = self.pos[2] + ((v7_z-self.pos[2]) * math.cos(self.rot) + (v7_x-self.pos[0]) * math.sin(self.rot))
        v7 = (v7_x2, v7_y, v7_z2)

        v8_x = self.pos[0]
        v8_y = self.pos[1] + self.height / 2
        v8_z = self.pos[2] + self.length / 2
        v8_x2 = self.pos[0] + ((v8_x - self.pos[0]) * math.cos(self.rot) - (v8_z - self.pos[2]) * math.sin(self.rot))
        v8_z2 = self.pos[2] + ((v8_z - self.pos[2]) * math.cos(self.rot) + (v8_x - self.pos[0]) * math.sin(self.rot))
        v8 = (v8_x2, v8_y, v8_z2)

        v9_x = self.pos[0]
        v9_y = self.pos[1] + self.height / 2
        v9_z = self.pos[2] + self.length / 2 + 1
        v9_x2 = self.pos[0] + ((v9_x - self.pos[0]) * math.cos(self.rot) - (v9_z - self.pos[2]) * math.sin(self.rot))
        v9_z2 = self.pos[2] + ((v9_z - self.pos[2]) * math.cos(self.rot) + (v9_x - self.pos[0]) * math.sin(self.rot))
        v9 = (v9_x2, v9_y, v9_z2)

        self.vertices = (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)

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

    def round_output(self, x):
        return "%.2f" % round(x,2)

    def print_obj(self):
        return "POS:(" + self.round_output(self.pos[0]) + "," + self.round_output(self.pos[1]) + "," + self.round_output(self.pos[2]) + ")  SIZE:" + self.round_output(self.width) + "," + self.round_output(self.height) + "," + self.round_output(self.length) + ") ROT:" + self.round_output(self.rot)


def draw_bounding_box(index, selected=False):
    glBegin(GL_LINES)
    if selected:
        glColor3fv((1, 1, 1))
    else:
        glColor3fv(boxes[index].color_value)
    for edge in box_edge_render_order:
        for vertex in edge:
            glVertex3fv(boxes[index].vertices[vertex])
    glEnd()
    if show_full_data and in_camera_mode:
        draw_text_3d(boxes[index].pos, boxes[index].object_type + str(index) + ": " + boxes[index].print_obj())
    else:
        draw_text_3d(boxes[index].pos, boxes[index].object_type + str(index))


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
    glColor3fv((0.35,0.5,0.25))
    for i in range(0,lines+1):
        glVertex3fv((plane_min + i*distance_between_lines,0,plane_min))
        glVertex3fv((plane_min + i*distance_between_lines,0,plane_max))
        glVertex3fv((plane_min, 0, plane_min + i*distance_between_lines))
        glVertex3fv((plane_max, 0, plane_min + i*distance_between_lines))
    glEnd()


def draw_background_image():
    # create projection matrix
    fov = math.radians(45.0)
    f = 1.0 / math.tan(fov / 2.0)
    zNear = 0.1
    zFar = 100.0
    aspect = render_size[0] / render_size[1]
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


def draw_2d_box(pos, size, bg_color):
    new_pos = (pos[0], render_size[1] - pos[1])

    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, render_size[0], render_size[1], 0, 0, 1)
    glBegin(GL_TRIANGLE_STRIP)
    glColor3f(bg_color[0], bg_color[1], bg_color[2])
    glVertex2f(new_pos[0], new_pos[1])
    glVertex2f(new_pos[0], new_pos[1] - size[1])
    glVertex2f(new_pos[0] + size[0], new_pos[1])
    glVertex2f(new_pos[0] + size[0], new_pos[1] - size[1])
    glEnd()

    glPopMatrix()

def draw_text(pos, text, color=(1,1,1), bg_color=(0,0,0)):

    draw_2d_box((pos[0] - text_border, pos[1] - 2 * text_border), (len(text) * 8 + 2 * text_border, 13 + text_border), bg_color)

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

def draw_text_3d(pos, text, color=(1,1,1)):

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


def instantiate_box(position=(0,0,0), rotation=0, size=(1,1,1), object_type="Car", color_value=(1,1,1)):
    new_box = BoundingBox(position=position, rotation=rotation, size=size,
                      object_type=object_type, color_value=color_value)
    global selected_box
    selected_box = len(boxes)
    boxes.append(new_box)


def mod_camera(x=0.0, y=0.0, z=0.0, rot_a=0.0, rot_b=0.0, fov=0.0):
    global camera_fov
    camera_pos[0] += x
    camera_pos[1] += y
    camera_pos[2] += z
    camera_rot[0] += rot_a
    camera_rot[1] += rot_b
    camera_fov += fov
    set_camera()

def round_output(x):
    return "%.3f" % round(x,3)

# Main program runtime loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            # Toggle mode
            if pygame.key.get_mods() & pygame.KMOD_ALT:
                in_camera_mode = not in_camera_mode
            # Toggle instruction visibility
            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                show_instructions = not show_instructions

            # ALT toggles between camera controls and box controls
            if in_camera_mode:
                # Camera reset
                if event.key == pygame.K_SPACE:
                    camera_pos = camera_pos_init.copy()
                    camera_rot = camera_rot_init.copy()
                    camera_fov = camera_fov_init
                    set_camera()

                # Camera translation
                elif event.key == pygame.K_w:
                    mod_camera(
                        z=camera_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_s:
                    mod_camera(
                        z=-camera_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_a:
                    mod_camera(
                        x=-camera_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_d:
                    mod_camera(
                        x=camera_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_q:
                    mod_camera(
                        y=camera_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_e:
                    mod_camera(
                        y=-camera_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))

                # Camera FOV manip
                elif event.key == pygame.K_r:
                    mod_camera(
                        fov=camera_mod_fov_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_f:
                    mod_camera(
                        fov=-camera_mod_fov_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))

                # Camera  rotation
                elif event.key == pygame.K_UP:
                    mod_camera(
                        rot_a = camera_rotation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_DOWN:
                    mod_camera(
                        rot_a = -camera_rotation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_LEFT:
                    mod_camera(
                        rot_b = camera_rotation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_RIGHT:
                    mod_camera(
                        rot_b = -camera_rotation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))

                # Print positioning
                elif event.key == pygame.K_p:
                    print("Camera position: " + str(camera_pos))
                    print("Camera rotation: " + str(camera_rot))
                    print("Camera FOV: " + str(camera_fov))
            elif in_place_mode:
                # Box type selection for creation
                if event.key == pygame.K_1:
                    instantiate_box(object_type=box_types[0][0], color_value=box_types[0][1], size=box_types[0][2])
                    box_blink_frame = 0
                    box_blink_state = True
                    in_place_mode = False
                elif event.key == pygame.K_2:
                    instantiate_box(object_type=box_types[1][0], color_value=box_types[1][1], size=box_types[1][2])
                    box_blink_frame = 0
                    box_blink_state = True
                    in_place_mode = False
                elif event.key == pygame.K_3:
                    instantiate_box(object_type=box_types[2][0], color_value=box_types[2][1], size=box_types[2][2])
                    box_blink_frame = 0
                    box_blink_state = True
                    in_place_mode = False
                elif (event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE):
                    in_place_mode = False
            else:
                # Box adjustment and deletion
                if event.key == pygame.K_RETURN:
                    in_place_mode = True
                    show_instructions = True
                elif len(boxes) == 0: # Program will crash if any below values called with 0 index and no boxes
                    None
                elif (event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE) and len(boxes) > 0:
                    boxes.pop(selected_box)
                    selected_box = min(selected_box,len(boxes)-1)  # clamp selected_box to usable range
                    box_blink_frame = 0
                    box_blink_state = True

                # Box selection
                elif (event.key == pygame.K_PAGEDOWN or event.key == pygame.K_z) and selected_box > 0 :  # Select Previous Box
                    selected_box -= 1
                    box_blink_frame = 0
                    box_blink_state = True
                elif (event.key == pygame.K_PAGEUP or event.key == pygame.K_x) and selected_box < len(boxes)-1:  # Select Next Box
                    selected_box += 1
                    box_blink_frame = 0
                    box_blink_state = True

                # Reset selected box to default values
                elif event.key == pygame.K_SPACE:
                    boxes[selected_box].reset()

                # Adjust selected box translation
                elif event.key == pygame.K_UP:
                    boxes[selected_box].mod_pos((0, 0, -box_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
                elif event.key == pygame.K_DOWN:
                    boxes[selected_box].mod_pos((0, 0, box_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)))
                elif event.key == pygame.K_LEFT:
                    boxes[selected_box].mod_pos((-box_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))
                elif event.key == pygame.K_RIGHT:
                    boxes[selected_box].mod_pos((box_translation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1), 0, 0))

                # Adjust selected box dimensions
                elif event.key == pygame.K_w:
                    boxes[selected_box].mod_height(box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_s and boxes[selected_box].height - (box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:
                    boxes[selected_box].mod_height(-box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_a and boxes[selected_box].width - (box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:  # Adjust Width of Selected Box
                    boxes[selected_box].mod_width(-box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_d:
                    boxes[selected_box].mod_width(box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_q and boxes[selected_box].length - (box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)) > 0:  # Adjust Length / Depth of Selected Box
                    boxes[selected_box].mod_length(-box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_e:
                    boxes[selected_box].mod_length(box_mod_dimension_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))

                # Adjust selected box rotation
                elif event.key == pygame.K_r:
                    boxes[selected_box].mod_rot(box_rotation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))
                elif event.key == pygame.K_f:
                    boxes[selected_box].mod_rot(-box_rotation_amount * (ctrl_multi if pygame.key.get_mods() & pygame.KMOD_CTRL else 1))

                # Print selected box
                elif event.key == pygame.K_p:
                    print(boxes[selected_box].print_obj())

    box_blink_frame += 1
    if box_blink_frame >= box_blink_speed:
        box_blink_state = not box_blink_state
        box_blink_frame = 0

    # Drawing code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_background_image()
    if in_camera_mode:
        draw_ground_plane_grid(100, 1)
        draw_axis()
    else:
        for index in range(0, len(boxes)):
            if selected_box == index and box_blink_state:
                draw_bounding_box(index, True)
            else:
                draw_bounding_box(index)
    if show_instructions:
        if not in_camera_mode:
            if in_place_mode:
                temp_string = "Cancel-[DEL/BACKSP] "
                for i in range(0,len(box_types)):
                    temp_string += " " + box_types[i][0] + "-[" + str(i+1) + "] "
                draw_text((2, 29), "[BOX ADJUST MODE] - [PLACING BOX]", bg_color=(0.8,0,0))
                draw_text((2, 15), temp_string, bg_color=(0.8, 0, 0))
            elif len(boxes) == 0:
                draw_text((2,29), "[BOX ADJUST MODE]")
                draw_text((2,15), "Mode-[ALT]  Hide-[SHIFT]  New-[ENTER]")
            else:
                draw_text((2,29), "[BOX ADJUST MODE]  Selected: " + boxes[selected_box].object_type + str(selected_box) + "  " + boxes[selected_box].print_obj())
                draw_text((2,15), "Mode-[ALT]  Hide-[SHIFT]  New-[ENTER]  Select-[Z,X]  Delete-[DEL/BACKSP]  Reset-[SPACE]")
                draw_text((2,2),  "Translate-[ARROWS]        Resize-[W,A,S,D,Q,E]       Rotate-[R,F]         PrintPos-[P] ")
        else:
            draw_text((2, 29), "[CAMERA ADJUST MODE]  POS:(" + str(round_output(camera_pos[0])) + "," + str(round_output(camera_pos[1])) + "," + str(round_output(camera_pos[2])) + ")   ROT:(" + str(round_output(camera_rot[0])) + "," + str(round_output(camera_rot[1])) + ")   FOV:" + str(round_output(camera_fov)))
            draw_text((2, 15), "Mode-[ALT]  Hide-[SHIFT]  Reset-[SPACE]")
            draw_text((2, 2),  "Translate-[W,A,S,D,E,Q]   Rotate-[ARROWS]   FOV-[R,F]  PrintPos-[P] ")

    # Update display and clock
    pygame.display.flip()
    clock.tick(30)  # 30 fps clock
