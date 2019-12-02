import tkinter
import tkinter.filedialog
from tkinter import *
from tkinter import Menu

trace = 0
traceText = 0


class CanvasEvents:
    def __init__(self, c):
        canvas = c
        canvas.pack()
        canvas.bind('<ButtonPress-1>', self.onStart)
        canvas.bind('<B1-Motion>', self.onGrow)
        canvas.bind('<Double-1>', self.onClear)
        canvas.bind('<ButtonPress-3>', self.onMove)
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
        # print("start X: ", self.start.x, " start Y:", self.start.y, "End X: ", event.x, "End Y: ", event.y)

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