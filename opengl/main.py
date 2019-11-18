from PIL import Image
from bounding_box import *
from render import *


# Sets the background image of the render process to the current frame
def change_frame():
    set_background_image(frames[current_frame])


# Adds a bounding box to the current frame based on several parameters
def instantiate_box(position, rotation, size, object_type, color_value):
    new_box = BoundingBox(position=position, rotation=rotation, size=size,
                      object_type=object_type, color_value=color_value)
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
BOX_TYPES = (("Car", (1, 0, 0), (2.0, 1.7, 5.0)),  # (name, default_color, default_size(w,h,l))
             ("Cyc", (0,1,0), (0.5,1.8,1.0)),
             ("Ped", (0,0,1), (0.5,1.7,0.5)))
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
    output = render_screen(boxes, BOX_TYPES, current_frame + 1, NUMBER_OF_FRAMES)  # Returns 0 unless a box needs to be instantiated
    if str(output).isdigit() and 0 < output < 10:
        instantiate_box(position=(0,0,0), rotation=0, object_type=BOX_TYPES[output - 1][0],
                        color_value=BOX_TYPES[output - 1][1], size=BOX_TYPES[output - 1][2])
    elif output == "f-" and current_frame >= 1:
        current_frame -= 1
        change_frame()
    elif output == "f+" and current_frame < (NUMBER_OF_FRAMES - 1):
        current_frame += 1
        change_frame()
    elif (not str(output).isdigit()) and output[0] == "f" and output[1] == "#":
        index = int(output[2:])
        if 0 < index <= NUMBER_OF_FRAMES:
            current_frame = index - 1
            change_frame()
    elif output == "output":
        output_string = ""
        for frame in frames:
            output_string += "["
            for i in range(0,len(boxes)):
                output_string += "[" + boxes[i].object_type + str(
                i) + "," + boxes[i].to_string_torch() + "]"
                if i < len(boxes)-1:
                    output_string += ","
            output_string += "],\n"
        f = open(OUTPUT_FILE_NAME,"w")
        f.write(output_string)
        f.close()
        print("Outputted to file " + OUTPUT_FILE_NAME)
