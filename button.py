class Button:
    def __init__(self, pos, text):
        self.ul = (pos[0], pos[1])
        self.text = text
        self.len = 8 * len(self.text) + 12
        self.triggered = False

    def get_ul(self):
        return self.ul

    def get_len(self):
        return self.len

    def get_text(self):
        return self.text

    def get_triggered(self):
        return self.triggered

    def is_triggered(self, mouse):
        x = self.ul[0] <= mouse[0] <= self.ul[0] + self.len - 1
        y = self.ul[1] <= mouse[1] <= self.ul[1] + 23
        if x and y:
            self.triggered = True
        return x and y

    def remove_trigger(self):
        self.triggered = False