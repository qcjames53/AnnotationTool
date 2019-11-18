from PIL import Image
from bounding_box import *
from render import *


# Sets the background image of the render process to the current frame
def change_frame():
    set_background_image(frames[current_frame])


# Adds a bounding box to the current frame based on several parameters
def instantiate_box(type, dimensions, location, rotation_y, color_value):
    # Constructor: (type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, color_value)
    new_box = BoundingBox(type, 0, 0, 0, [0,0,0,0], dimensions, location, rotation_y, color_value)
    global selected_box
    selected_box = len(boxes)
    boxes.append(new_box)


# Loads a given number of frames following a naming convention of type "*.png" where * is replaced by integers from
# 000001 to number_of_frames (each integer padded to 6 decimal places)
def setup_frames(number_of_frames, naming_convention):
    print("Loading " + str(number_of_frames) + " images of type " + naming_convention)
    for i in range(1,number_of_frames+1):
        index = str(i)
        while len(index) < 6:  # Padding
            index = "0" + index
        PIL_image = Image.open(naming_convention.replace("*",index))
        frames.append(PIL_image)
    print ("Loading complete")


# Global constants
BOX_TYPES = (("Car", [1.7, 2.0, 5.0], (1, 0, 0)),  # (name, dimensions(h,w,l), color_value)
             ("Cyclist", [1.8,0.5,2.0], (0,1,0)),
             ("Pedestrian", [1.7,0.5,0.5], (0,0,1)))
NUMBER_OF_FRAMES = 100
OUTPUT_FILE_NAME = "output.txt"

# Global variables
current_frame = 0
frames = []
boxes = []


# Initial setup
setup_frames(NUMBER_OF_FRAMES, "images/img*.jpg")
change_frame()

# Main program loop. Limited to 30fps by render system
while True:
    output = render_screen(boxes, BOX_TYPES, current_frame + 1, NUMBER_OF_FRAMES)
    if output.find("b#") != -1:
        index = int(output[2:])
        if 0 < index <= len(BOX_TYPES):
            instantiate_box(BOX_TYPES[index - 1][0], BOX_TYPES[index - 1][1], [0,0,0], 0, BOX_TYPES[index - 1][2])
    elif output.find("f-") != -1 and current_frame >= 1:
        current_frame -= 1
        change_frame()
    elif output.find("f+") != -1 and current_frame < (NUMBER_OF_FRAMES - 1):
        current_frame += 1
        change_frame()
    elif output.find("f#") != -1:
        index = int(output[2:])
        if 0 < index <= NUMBER_OF_FRAMES:
            current_frame = index - 1
            change_frame()
    elif output == "output":
        output_string = ""
        for box in boxes:
            output_string += box.to_string()
        f = open(OUTPUT_FILE_NAME,"w")
        f.write(output_string)
        f.close()
        print("Outputted to file " + OUTPUT_FILE_NAME)
