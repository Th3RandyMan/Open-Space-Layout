import numpy as np
from enum import IntEnum

# Lab room 40ft by 19.2ft (400,192)ft/10
# Desk 58inches by 23inches (58, 23)inches
# 

class Rotation(IntEnum):
    """
    Enum class for rotation. Objects facing up will have reserved space defined above the object.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    TBI = 4

    def __str__(self):
        if self.value == Rotation.UP:
            return "UP"
        elif self.value == Rotation.RIGHT:
            return "RIGHT"
        elif self.value == Rotation.DOWN:
            return "DOWN"
        elif self.value == Rotation.LEFT:
            return "LEFT"
        else:
            raise AttributeError("UNKNOWN")
    

class Object:
    """
    Abstract class for all objects in the storage. Width, depth, reserved space and id are 
    common for all objects. Position of objects will always be referenced from the bottom 
    right corner of the object.
    """
    def __init__(self, width, depth, reserved_space, id, uid, name):
        self.width = width
        self.depth = depth
        self.reserved_space = reserved_space
        self.id = id
        self.rid = -1   #reference id
        self.uid = uid
        self.name = name    #maybe remove? change to seperate objects?
        self.rotation = Rotation.TBI

    def __str__(self):
        return f"{self.name} | Width: {self.width}, Depth: {self.depth}, Reserved Space: {self.reserved_space}, ID: {self.id}"
    
    def set_position(self, x, y, rotation):
        self.x = x
        self.y = y
        self.rotation = rotation
    
    def move(self, x, y, rotation):
        self.x += x
        self.y += y
        self.rotation = rotation

# class Desk(Object):
#     """
#     Class for desk objects. Inherits from Object class.
#     """
#     def __init__(self, width, depth, reserved_space, id):
#         super().__init__(width, depth, reserved_space, id)
#         self.name = "Desk"

#     def __str__(self):
#         return f"Desk | {super().__str__()}"

# Matrix for unique id of object
# Matrix for if space is taken (0,1)
# Matrix for open space (without reserved space)
        
# Solution space will be a list of lists. First list iterated through by id, second list iterated by reference id. 
    # Reference id will be assigned based on distance from the bottom left corner of the room. Euclidean distance because of calculating distance between solutions.

# Conflict Types:
        # 1. Static object hit. Dynamic object will move as much as it can.
        # 2. Dynamic object hit. Dynamic objects will have uniform distribution of who gets the space.

# How to find conflicts:
        # 1. Use prime number unique ids, open space will have 1s, movement will multiply new space and divide previous space.
            # Just need to check if prime number in each new position.
        # 2. Use a list of conflict objects. Each object consists of location (x,y) and list of uid with conflicts. If single
            # object is in list, no conflicts.
        


class Room:
    """
    
    """

    def __init__(self, width, height, name):
        self.width = width
        self.height = height
        self.name = name

        self.space = np.zeros((width, height), dtype=int)
        self.objects = []

    def add_object(self, object, x, y, rotation):
        """
        Adds object to the room.
        """
        if self.fits(object, x, y, rotation):
            object.set_position(x, y, rotation)
            self.objects.append(object)
            return True
        else:
            return False

    def fits(self, object, x, y, rotation):
        """
        Checks if object fits in the room.
        """
        if rotation == Rotation.UP:
            return x + object.width <= self.width and y + object.depth <= self.depth
        elif rotation == Rotation.RIGHT:
            return x + object.depth <= self.width and y + object.width <= self.depth
        elif rotation == Rotation.DOWN:
            return x + object.width <= self.width and y + object.depth <= self.depth
        elif rotation == Rotation.LEFT:
            return x + object.depth <= self.width and y + object.width <= self.depth
        else:
            return False

    def __str__(self):
        return f"Room | Width: {self.width}, Height: {self.height}, Number of Objects: {len(self.objects)}\n" \
                f"Objects: {[self.objects[i].name for i in range(len(self.objects))]}"