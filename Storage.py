from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from enum import IntEnum
from sympy import nextprime
from collections import Counter
from functools import reduce

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
            return "UNKNOWN"
    

class Object: # May want to change name to not confuse with object class (lower case o)
    """
    Abstract class for all objects in the storage. Width, depth, reserved space and id are 
    common for all objects. Position of objects will always be referenced from the bottom 
    left corner of the object.
    """
    def __init__(self, width: int, depth: int, reserved_space: int, name: str, id: int=-1, rotation: Rotation = Rotation.TBD, moveable: bool = True, rotatable: bool = True):
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
    
    def shape_index(self):
        return f"{self.width}_{self.depth}_{self.reserved_space}_{self.rotatable}"

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
    fobj = None # objective function
    X = [] # location in solution space

    def __init__(self, width, height, name):
        self.width = width
        self.height = height
        self.name = name
        self.current_uid = 1 # must be prime numbers
        #self.current_rid = 0 # reference id for objects of the same shape

        # Maybe change to all one array?
        self.uid_space = np.ones((width, height), dtype=int)   #unique id space for moving objects around
        self.taken_space = np.zeros((width, height), dtype=bool)#space taken by objects, including reserved space
        self.open_space = np.zeros((width, height), dtype=bool) #space that is not taken by objects, not including reserved space
        self.objects = defaultdict(Object) #objects in the room, key is the object uid
        self.moveable_shapes = defaultdict(list) #list of objects with the same shape, key is the shape index
        #self.rid_list = list[int] #list of reference ids for objects of the same shape

    def get_inventory(self) -> list[str]:
        """
        """
        obj_list = [self.objects[uid].name for uid in self.objects.keys()]
        duplicates = Counter(obj_list)
        return [f"{k}: {v}" for k, v in duplicates.items()]

    def __str__(self):
        return f"Room | Width: {self.width}, Height: {self.height}, Number of Objects: {len(self.objects)}\n" \
                f"Objects: {self.get_inventory()}"


    def next_uid(self) -> int:
        """
        Returns the next prime number for unique id.
        """
        self.current_uid = nextprime(self.current_uid)
        return self.current_uid
    

    def add_object(self, object, x, y, rotation) -> bool:
        """
        Adds object to the room.
        """
        if self._fits(object, x, y, rotation):
            object.set_position(x, y, rotation)
            self._add_object_to_space(object)
            return True
        else:
            return False
        

    def _fits(self, object: Object, x: int, y: int, rotation: Rotation) -> bool:
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
        

    def _add_object_to_space(self, object: Object) -> None:
        """
        Updates the space taken by the object in the room.
        """
        object.uid = self.next_uid()
        self.objects[object.uid] = object
        if object.moveable:
            self.moveable_shapes[object.shape_index()].append(object.uid)

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
        
    def uid_map(self) -> np.ndarray:
        """
        Returns the uid space.
        """
        return self.uid_space.transpose()
    
    def open_map(self) -> np.ndarray:
        """
        Returns the open space.
        """
        return self.open_space.transpose()
    
    def get_X(self) -> list[np.ndarray]:
        """
        Get solution space X.
        """
        X = []

        for key in self.moveable_shapes.keys():
            # Sort the objects by distance from origin
            dists = []
            for uid in self.moveable_shapes[key]:
                obj = self.objects[uid]
                dists.append((norm([obj.x, obj.y]), uid))
            dists.sort()    # sort by distance from origin
            
            self.moveable_shapes[key] = [uid for _, uid in dists] #update the list of uids to be sorted by distance from origin
            
            if key[-4:] == "True": #rotatable
                xi = np.zeros((len(dists),3)) #x, y, rotation
                for i, uid in enumerate(self.moveable_shapes[key]):
                    obj = self.objects[uid]
                    xi[i] = [obj.x, obj.y, obj.rotation]
            else:   #not rotatable
                xi = np.zeros((len(dists),2)) #x, y
                for i, uid in enumerate(self.moveable_shapes[key]):
                    obj = self.objects[uid]
                    xi[i] = [obj.x, obj.y]
            X.append(xi)
        
        self.X = X
        return X
    
    # def set_X(self, X: list[np.ndarray]):
    #     """
    #     Set solution space X.
    #     """
    #     X_round = [np.round(xi) for xi in X]
    #     valid, new_uid_space = self.check_space(X_round)
        # np.ones((self.width, self.height), dtype=int)#self.uid_space
        # for i, key in enumerate(self.moveable_shapes.keys()):
        #     for j, uid in enumerate(self.moveable_shapes[key]):
        #         obj = self.objects[uid]
        #         X_round[i][j] # x, y, rotation adjustments


        # for i, key in enumerate(self.moveable_shapes.keys()):
        #     for j, uid in enumerate(self.moveable_shapes[key]):
        #         obj = self.objects[uid]
        #         obj.x = X[i][j][0]
        #         obj.y = X[i][j][1]
        #         if X[i].shape[1] == 3:
        #             obj.rotation = X[i][j][2]
    
    def set_X(self, X_float: list[np.ndarray]) -> bool:
        """
        Check for collisions in the new space.
        """
        X = [np.round(xi) for xi in X_float]
        new_uid_space = np.ones((self.width, self.height), dtype=int)
        conflicts = set()
        uid_to_X = defaultdict(tuple)

        # Create new uid space and check for conflicts
        for i, key in enumerate(self.moveable_shapes.keys()):
            for j, uid in enumerate(self.moveable_shapes[key]):
                obj = self.objects[uid]
                uid_to_X[uid] = (i,j)
                x, y = X[i][j][0] + obj.x, X[i][j][1] + obj.y
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0

                rotation = (X[i][j][2] + obj.rotation)%4 if X[i].shape[1] == 3 else obj.rotation

                if rotation == Rotation.UP:
                    if x + obj.width > self.width:
                        x = self.width - obj.width
                    if y + obj.depth + obj.reserved_space > self.height:
                        y = self.height - obj.depth - obj.reserved_space
                    new_uid_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] *= obj.uid
                    conf = np.where(new_uid_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] > obj.uid)

                elif obj.rotation == Rotation.RIGHT:
                    if x + obj.depth + obj.reserved_space > self.width:
                        x = self.width - obj.depth - obj.reserved_space
                    if y + obj.width > self.height:
                        y = self.height - obj.width
                    new_uid_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] *= obj.uid
                    conf = np.where(new_uid_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] > obj.uid)                 

                elif obj.rotation == Rotation.DOWN:
                    if x + obj.width > self.width:
                        x = self.width - obj.width
                    if y + obj.depth > self.height:
                        y = self.height - obj.depth
                    new_uid_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] *= obj.uid
                    conf = np.where(new_uid_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] > obj.uid)

                elif obj.rotation == Rotation.LEFT:
                    if x + obj.depth > self.width:
                        x = self.width - obj.depth
                    if y + obj.width > self.height:
                        y = self.height - obj.width
                    new_uid_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] *= obj.uid
                    conf = np.where(new_uid_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] > obj.uid)

                else:
                    raise AttributeError("UNKNOWN")
                
                if conf[0].size > 0:    # If there are conflicts, add to set
                    conflicts.append(conf)
                    #uid_to_X[uid] = (i,j) # store the uid and its position in X

        # Resolve conflicts
                    # NOTES: IF OBJECT HASN'T MOVED, GIVE IT THE POSITION
                    # IF ALL OBJECTS HAVE MOVED, EITHER UNIFORM DISTRIBUTION OR GIVE WEIGHT TO CLOSER OBJECTS
        for conf in conflicts:
            uid_list = [self._factorize(num) for num in new_uid_space[conf]] # Get the uids fighting for the space
            #probability = []
            #prob = 1/len(uid_list)
            for uid in uid_list:
                i, j = uid_to_X[uid]
                x, y = X[i][j][0], X[i][j][1]
                # Check if object hasn't moved. Give it the position if it hasn't moved
                if x == 0 and y == 0:
                    pass
            #new_uid_space[conf] = 0



        return True # Change this

    def _factorize(self, n: int) -> list[int]:
        """
        Returns the factors of n.
        :param n: int to factorize
        :return: list of factors
        """
        if n <= 1:
            return []
        
        factors = set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
        return factors.pop()

    def evaluate(self) -> float:
        """
        Evaluate the objective function.
        """
        # Get value of open space here!
        fobj = 0.0
        return fobj




if __name__ == "__main__":
    # Test
    from matplotlib import pyplot as plt
    room = Room(100, 100, "Test Room")

    table1 = Object(10, 10, 5, 1, "Table") 
    couch1 = Object(20, 10, 5, 2, "Couch")
    door1 = Object(10, 0, 10, 3, "Door")

    room.add_object(table1, 10, 10, Rotation.UP)
    room.add_object(couch1, 40, 40, Rotation.LEFT)
    room.add_object(door1, 80, 0, Rotation.UP)

    # print(table1)
    # print(couch1)
    # print(door1)
    print(room)
    print(f"Solution: {room.get_X()}")

    plt.subplot(1, 2, 1)
    plt.imshow(room.uid_map(), origin='lower')
    plt.title("UID Space")

    plt.subplot(1, 2, 2)
    plt.imshow(room.open_map(), origin='lower')
    plt.title("Open Space")

    plt.show()