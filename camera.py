class Camera:
    '''
    A class to store camera parameters for the render function.
    '''
    def __init__(self, pos, rot, fov, ncp, fcp):
        self.pos = pos.copy()
        self.pos_init = pos.copy()
        self.rot = rot.copy()
        self.rot_init = rot.copy()
        self.fov = fov
        self.fov_init = fov
        self.ncp = ncp
        self.fcp = fcp

    def move(self, x=0.0, y=0.0, z=0.0, rot_a=0.0, rot_b=0.0, fov=0.0):
        global camera_fov
        self.pos[0] += x
        self.pos[1] += y
        self.pos[2] += z
        self.rot[0] += rot_a
        self.rot[1] += rot_b
        self.fov += fov

    def reset(self):
        self.pos = self.pos_init.copy()
        self.rot = self.rot_init.copy()
        self.fov = self.fov_init

    def get_pos_copy(self):
        return self.pos.copy()