from collections import defaultdict
import numpy as np
from enum import IntEnum
from sympy import nextprime


class Rotation(IntEnum):
    """
    Enum class for rotation. Objects facing up will have reserved space defined above the object.
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    TBD = 4 #To Be Declared

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
    left corner of the object.
    """
    def __init__(self, width: int, depth: int, reserved_space: int, id, name: str, rotation: Rotation = Rotation.TBD, moveable: bool = True, rotatable: bool = True):
        self.width = width
        self.depth = depth
        self.reserved_space = reserved_space
        self.id = id    #id for objects of the same shape
        self.rid = -1   #reference id for seperating objects of the same shape
        self.uid = -1   #unique id for the object, make a prime number
        self.name = name    #maybe remove? change to seperate objects?
        self.rotation = rotation
        self.moveable = moveable
        self.rotatable = rotatable

    def __str__(self):
        return f"{self.name} | Width: {self.width}, Depth: {self.depth}, Reserved Space: {self.reserved_space}, Rotation: {self.rotation}\n" \
                f"\tID: {self.id}, RID: {self.rid}, UID: {self.uid}"
    
    def set_position(self, x, y, rotation):
        self.x = x
        self.y = y
        self.rotation = rotation
    
    def move(self, x, y, rotation):
        if(self.moveable):
            self.x += x
            self.y += y
            self.rotation = rotation
        else:
            raise AttributeError("Object is not moveable")


class Room:
    """
    
    """

    def __init__(self, width, height, name):
        self.width = width
        self.height = height
        self.name = name
        self.current_uid = 1 # must be prime numbers

        # Maybe change to all one array?
        self.uid_space = np.zeros((width, height), dtype=int)   #unique id space for moving objects around
        self.taken_space = np.zeros((width, height), dtype=bool)#space taken by objects, including reserved space
        self.open_space = np.zeros((width, height), dtype=bool) #space that is not taken by objects, not including reserved space
        self.objects = defaultdict(object) #objects in the room, key is the object uid


    def __str__(self):
        return f"Room | Width: {self.width}, Height: {self.height}, Number of Objects: {len(self.objects)}\n" \
                f"Objects: {[self.objects[i].name for i in range(len(self.objects))]}"


    def next_uid(self):
        """
        Returns the next prime number for unique id.
        """
        self.current_uid = nextprime(self.current_uid)
        return self.current_uid
    

    def add_object(self, object, x, y, rotation):
        """
        Adds object to the room.
        """
        if self.fits(object, x, y, rotation):
            object.set_position(x, y, rotation)
            self.add_object_to_space(object)
            return True
        else:
            return False
        

    def fits(self, object: Object, x: int, y: int, rotation: Rotation):
        """
        Checks if object fits in the room at position x, y with rotation.
        """
        if(rotation == Rotation.TBD):
            return False
        if(x < 0 or y < 0 or y > self.height - 1 or x > self.width - 1): #check if object is outside of room
            return False
                
        depth = object.depth + object.reserved_space
        # Check direction of object, and if it fits in the room. Iterate through the space the object will take up.
        if (rotation == Rotation.UP) and (x + object.width < self.width and y + depth < self.height):
            for i in range(object.width):
                for j in range(depth):
                    if self.taken_space[x + i][y + j]:
                        return False
        elif (rotation == Rotation.RIGHT) and (x + depth < self.width and y + object.width < self.height):
            for i in range(depth):
                for j in range(object.width):
                    if self.taken_space[x + i][y + j]:
                        return False
        elif (rotation == Rotation.DOWN) and (x + object.width < self.width and y + object.depth < self.height) and (y - object.reserved_space >= 0):
            for i in range(object.width):
                for j in range(-object.reserved_space, object.depth):
                    if self.taken_space[x + i][y + j]:
                        return False
        elif (rotation == Rotation.LEFT) and (x + object.depth < self.width and y + object.width < self.height) and (x - object.reserved_space >= 0):
            for i in range(-object.reserved_space, object.depth):
                for j in range(object.width):
                    if self.taken_space[x + i][y + j]:
                        return False
        # elif (rotation == Rotation.DOWN) and (x + object.width < self.width and y + depth < self.height):
        #     for i in range(object.width):
        #         for j in range(depth):
        #             if self.taken_space[x + i][y + j]:
        #                 return False
        # elif (rotation == Rotation.LEFT) and (x + depth < self.width and y + object.width < self.height):
        #     for i in range(depth):
        #         for j in range(object.width):
        #             if self.taken_space[x + i][y + j]:
        #                 return False
        else:
            return False    # Doesn't fit in the room
        
        return True
        

    def add_object_to_space(self, object: Object):
        """
        Updates the space taken by the object in the room.
        """
        object.uid = self.next_uid()
        self.objects[object.uid] = object

        if object.rotation == Rotation.UP:
            self.uid_space[object.x : object.x + object.width, object.y : object.y + object.depth + object.reserved_space] = object.uid
            self.taken_space[object.x : object.x + object.width, object.y : object.y + object.depth + object.reserved_space] = True
            self.open_space[object.x : object.x + object.width, object.y : object.y + object.depth] = True
        elif object.rotation == Rotation.RIGHT:
            self.uid_space[object.x : object.x + object.depth + object.reserved_space, object.y : object.y + object.width] = object.uid
            self.taken_space[object.x : object.x + object.depth + object.reserved_space, object.y : object.y + object.width] = True
            self.open_space[object.x : object.x + object.depth, object.y : object.y + object.width] = True
        elif object.rotation == Rotation.DOWN:
            self.uid_space[object.x : object.x + object.width, object.y - object.reserved_space : object.y + object.depth] = object.uid
            self.taken_space[object.x : object.x + object.width, object.y - object.reserved_space : object.y + object.depth] = True
            self.open_space[object.x : object.x + object.width, object.y : object.y + object.depth] = True
        elif object.rotation == Rotation.LEFT:
            self.uid_space[object.x - object.reserved_space : object.x + object.depth, object.y : object.y + object.width] = object.uid
            self.taken_space[object.x - object.reserved_space : object.x + object.depth, object.y : object.y + object.width] = True
            self.open_space[object.x : object.x + object.depth, object.y : object.y + object.width] = True
        else:
            raise AttributeError("UNKNOWN")



if __name__ == "__main__":
    # Test
    room = Room(100, 100, "Test Room")
    print(room)

    table1 = Object(10, 10, 0, 1, 1, "Table")
    couch1 = Object(10, 10, 0, 2, 2, "Couch")
    print(table1)
    print(couch1)