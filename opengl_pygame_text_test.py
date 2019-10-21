#!/usr/bin/env python

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pygame

ESCAPE = '\033'
window = 0
texture = 0

A_TEX_NUMBER = None
B_TEX_NUMBER = None

def GenTextureForText(text):
    font = pygame.font.Font(None, 64)
    textSurface = font.render(text, True, (255,255,255,255),
                              (0,0,0,255))
    ix, iy = textSurface.get_width(), textSurface.get_height()
    image = pygame.image.tostring(textSurface, "RGBX", True)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    i = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, i)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    return i

def InitGL(Width, Height):
    global A_TEX_NUMBER, B_TEX_NUMBER
    pygame.init()
    A_TEX_NUMBER = GenTextureForText("a")
    B_TEX_NUMBER = GenTextureForText("b")
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    done = 1

def DrawGLScene():

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
  glLoadIdentity()
  glTranslatef(0.0,0.0,-10.0)
  glBindTexture(GL_TEXTURE_2D, B_TEX_NUMBER)
  glBegin(GL_QUADS)
  glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0,  1.0)
  glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0,  1.0)
  glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0,  1.0)
  glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0,  1.0)
  glEnd()
  glutSwapBuffers()

def keyPressed(*args):
    if args[0] == ESCAPE:
      glutDestroyWindow(window)
      sys.exit()

def main():
  global window
  glutInit("")
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
  glutInitWindowSize(640, 480)
  glutInitWindowPosition(0, 0)
  window = glutCreateWindow("Jeff Molofee's GL Code Tutorial ... NeHe '99")
  glutDisplayFunc(DrawGLScene)
  glutIdleFunc(DrawGLScene)
  glutKeyboardFunc(keyPressed)
  InitGL(640, 480)
  glutMainLoop()
print("Hit ESC key to quit.")
main()