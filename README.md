# AnnotationTool

![Screenshot](https://qcjames53.github.io/AnnotationTool/2019-12-02.png)

AnnotationTool is a Python 3.6 library that allows for the manual 3D annotation of images in arbitrary reference planes. The program allows for the tracking of arbitrary types of objects, outputting each object's position in a format consistent with the [KITTI data set](http://www.cvlibs.net/datasets/kitti/) when requested.

This library is actively under development. Future plans revolve around implementing this tool as a method for visualizing and training the [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN) library, from Michigan State's computer vision lab.

## Installation
The library currently requires [Python 3.6](https://www.python.org/downloads/release/python-368/) to help mitigate potential issues with integrating [M3D-RPN](https://github.com/garrickbrazil/M3D-RPN) in the future.

##### Required installations:
```
pip install [both OpenGL .whl files]
pip install numpy
pip install pillow
pip install pygame
```

If a different version of Python is required, the necessary wheel files can be sourced from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl.

## Usage
AnnotationTool can be run via `python main.py`. It is preconfigured to load a set amount of low-filesize .jpg images from the `/images` directory. Available keyboard controls are displayed.

Default parameters can be tweaked in the `main.py` file, allowing for arbitrary bounding box types and selection of differing images. In the image filename, '*' will be replaced by a six-digit number starting at '000000' and ending at the selected 'NUMBER-OF-FRAMES'.
