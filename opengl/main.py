from bounding_box import *
from render import *


def instantiate_box(position=(0,0,0), rotation=0, size=(1,1,1), object_type="Car", color_value=(1,1,1)):
    new_box = BoundingBox(position=position, rotation=rotation, size=size,
                      object_type=object_type, color_value=color_value)
    global selected_box
    selected_box = len(boxes)
    boxes.append(new_box)


boxes = []  # An array of all stored bounding boxes
box_types = (("Car", (1,0,0), (2.0,1.7,5.0)),  # (name, default_color, default_size(w,h,l))
             ("Cyc", (0,1,0), (0.5,1.8,1.0)),
             ("Ped", (0,0,1), (0.5,1.7,0.5)),
             ("Tre", (1,1,0), (9.0, 20,9.0)))
set_background_image("test_image_measures.jpg")


# Main program loop. Limited to 30fps by render system
while True:
    output = render_screen(boxes, box_types)  # Returns 0 unless a box needs to be instantiated
    if output != 0:
        instantiate_box(object_type=box_types[output-1][0], color_value=box_types[output-1][1], size=box_types[output-1][2])