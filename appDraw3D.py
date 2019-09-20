import tkinter
import tkinter.filedialog
from tkinter import *
from tkinter import Menu
from itertools import cycle
from random import randint

trace = 0
traceText = 0

COLORS = cycle(['red', 'blue', 'green', 'magenta', 'yellow'])
checkRelease = True


class PerspectiveCube:
    def __init__(self, canvas, width, height, locX, locY):
        self.canvas = canvas
        self.polygons = []
        self.BOX_SIZE = width/2
        self.h = height
        self.w = width
        self.locX = locX
        self.locY = locY
        color = next(COLORS)
        for _ in range(4):
            p = canvas.create_polygon(0,0,0,0,0,0, outline=color, fill='', width=4)
            self.canvas.tag_bind(p, "<B1-Motion>", self._on_clickndrag)
            self.polygons.append(p)

        self.update_screen(locX, locY)  # initial point

    def _on_clickndrag(self, event):
        self.update_screen(event.x, event.y)
        # checkRelease = True

    def update_screen(self, x=None, y=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        half_width = self.locX
        half_height = self.locY-25
        x1 = self.x-self.BOX_SIZE
        y1 = self.y-self.BOX_SIZE
        x2 = self.x+self.BOX_SIZE
        y2 = self.y+self.BOX_SIZE
        x3 = (half_width+x1)/2
        y3 = (half_height+y2)/2
        x4 = (half_width+x1+self.BOX_SIZE*2)/2
        y4 = (half_height+y2)/2
        x5 = (half_width+x1)/2
        y5 = (half_height+y1)/2
        x6 = (half_width+x1)/2
        y6 = (half_height+y2)/2
        x7 = (half_width+x1+self.BOX_SIZE*2)/2
        y7 = (half_height+y1)/2

        self.canvas.coords(self.polygons[0], x3, y3, x4, y4, x2, y2, x2 - self.BOX_SIZE*2, y2)
        self.canvas.coords(self.polygons[1], x5, y5, x6, y6, x2 - self.BOX_SIZE*2, y2, x1, y1)
        self.canvas.coords(self.polygons[2], x5, y5, x1, y1, x1+self.BOX_SIZE*2, y1, x7, y7)
        self.canvas.coords(self.polygons[3], x2, y2, x1+self.BOX_SIZE*2, y1, x7, y7, x4, y4)


class CanvasEvents:
    def __init__(self, c):
        canvas = c
        canvas.pack()
        canvas.bind('<ButtonPress-1>', self.onStart)
        canvas.bind('<B1-Motion>', self.onGrow)
        canvas.bind('<Double-1>', self.onClear)
        canvas.bind('<ButtonPress-3>', self.onMove)
        # canvas.bind('<ButtonRelease-1>', self.onRelease)
        self.canvas = canvas
        self.drawn = None
        self.drawnText1 = None
        self.drawnText2 = None
        self.drawnText3 = None

    def onStart(self, event):
        self.start = event
        self.drawn = None
        self.drawnText1 = None
        self.drawnText2 = None
        self.drawnText3 = None

    def onGrow(self, event):
        canvas = event.widget
        if self.drawn: canvas.delete(self.drawn)
        objectId = canvas.create_rectangle(self.start.x, self.start.y, event.x, event.y, outline='red', tag="box")

        if self.drawnText1:
            canvas.delete(self.drawnText1)
            canvas.delete(self.drawnText2)
            canvas.delete(self.drawnText3)

        text1 = str(self.start.x) + ", " + str(self.start.y)
        textId1 = self.canvas.create_text(self.start.x, self.start.y, fill="darkblue", font="Times 20 italic bold",
                                text=text1, tag="box")
        text2 = str(self.start.x) + ", " + str(event.y)
        textId2 = self.canvas.create_text(self.start.x, event.y, fill="darkblue", font="Times 20 italic bold",
                                text=text2, tag="box")

        text3 = str(event.x) + ", " + str(self.start.y)
        textId3 = self.canvas.create_text(event.x, self.start.y, fill="darkblue", font="Times 20 italic bold",
                                          text=text3, tag="box")

        text4 = str(event.x) + ", " + str(event.y)
        textId4 = self.canvas.create_text(event.x, event.y, fill="darkblue", font="Times 20 italic bold",
                                          text=text4, tag="box")
        if trace:
            print
        objectId
        self.drawn = objectId

        if traceText:
            print
        textId2
        textId3
        textId4
        self.drawnText1 = textId2
        self.drawnText2 = textId3
        self.drawnText3 = textId4

    def onClear(self, event):
        event.widget.delete('box')

    def onMove(self, event):
        if self.drawn:
            if trace: print
            self.drawn
            canvas = event.widget
            diffX, diffY = (event.x - self.start.x), (event.y - self.start.y)
            canvas.move(self.drawn, diffX, diffY)
            self.start = event

    def onRelease(self, event):
        width = event.x - self.start.x
        height = event.y - self.start.y
        print("width: ", width, " height: ", height)

        locX = self.start.x + (width / 2)
        locY = self.start.y + (height / 2)
        PerspectiveCube(self.canvas, width, height, locX, locY)


def main():
    window = tkinter.Tk()
    window.title("Annotation Tool")
    window.geometry('350x200')

    def clicked():
        f = tkinter.filedialog.askopenfilename(
            parent=window,
            title='Choose file',
            filetypes=[('png images', '.png'),
                       ('gif images', '.gif')]
        )

        new_window = tkinter.Toplevel(window)
        image = PhotoImage(file=f)
        canvas = Canvas(new_window, width=1800, height=1800)
        canvas.pack()
        canvas.create_image(0, 0, image=image, anchor=NW)
        CanvasEvents(canvas)
        menu = Menu(new_window)

        new_item = Menu(menu)

        new_item.add_command(label='2D')

        new_item.add_separator()

        new_item.add_command(label='3D')

        menu.add_cascade(label='Tool', menu=new_item)

        new_window.config(menu=menu)
        new_window.mainloop()

    b1 = tkinter.Button(window, text='Choose File', command=clicked)
    b1.pack(fill='x')

    window.mainloop()


if __name__ == '__main__':
    main()