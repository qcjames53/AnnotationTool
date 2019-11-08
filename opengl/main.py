from PIL import Image
from bounding_box import *
from render import *


def instantiate_box(position, rotation, size, object_type, color_value):
    new_box = BoundingBox(position=position, rotation=rotation, size=size,
                      object_type=object_type, color_value=color_value)
    global selected_box
    selected_box = len(frames[current_frame][0])
    frames[current_frame][0].append(new_box)


def setup_frames(number_of_frames, naming_convention):
    print ("Loading " + str(number_of_frames) + " images of type " + naming_convention)
    for i in range(1,number_of_frames+1):
        container_list = []  # A list to hold boxes and the frame, along with any other data for the future
        boxes = []  # A list of all stored bounding boxes
        index = str(i)
        while len(index) < 6:  # Padding
            index = "0" + index
        PIL_image = Image.open(naming_convention.replace("*",index))

        container_list.append(boxes)
        container_list.append(PIL_image)
        frames.append(container_list)
    print ("Loading complete")


def change_frame():
    set_background_image(frames[current_frame][1])


frames = []
box_types = (("Car", (1,0,0), (2.0,1.7,5.0)),  # (name, default_color, default_size(w,h,l))
             ("Cyc", (0,1,0), (0.5,1.8,1.0)),
             ("Ped", (0,0,1), (0.5,1.7,0.5)),
             ("Tre", (1,1,0), (9.0, 20,9.0)))
current_frame = 0
number_of_images = 100
setup_frames(number_of_images, "images/img*.jpg")
change_frame()

# Main program loop. Limited to 30fps by render system
while True:
    output = render_screen(frames[current_frame][0], box_types, current_frame+1, number_of_images)  # Returns 0 unless a box needs to be instantiated
    if str(output).isdigit() and 0 < output < 10:
        instantiate_box(position=(0,0,0), rotation=0, object_type=box_types[output-1][0],
                        color_value=box_types[output-1][1], size=box_types[output-1][2])
    elif output == "f-" and current_frame > 1:
        current_frame -= 1
        change_frame()
    elif output == "f+" and current_frame < (number_of_images - 1):
        current_frame += 1
        change_frame()
    elif (not str(output).isdigit()) and output[0] == "f" and output[1] == "#":
        index = int(output[2:])
        if 0 < index <= number_of_images:
            current_frame = index - 1
            change_frame()