from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from enum import IntEnum
from sympy import nextprime
from collections import Counter
from functools import reduce
from matplotlib import pyplot as plt
from skimage.measure import label



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
    def __init__(self, width: int, depth: int, reserved_space: int, name: str, x: int=-1, y: int=-1, id: int=-1, rotation: Rotation = Rotation.TBD, moveable: bool = True, rotatable: bool = True):
        """
        :param width: Width of the object.
        :param depth: Depth of the object.
        :param reserved_space: Space reserved for the object. This is the space above the object when the object is facing up.
        :param name: Name of the object.
        :param x: x position of the object. Default is -1.
        :param y: y position of the object. Default is -1.
        :param id: Id of the object. Default is -1.
        :param rotation: Rotation of the object. Default is TBD.
        :param moveable: If the object is moveable. Default is True.
        :param rotatable: If the object is rotatable. Default is True.
        """
        self.width = width
        self.depth = depth
        self.reserved_space = reserved_space
        self.x = x
        self.y = y
        self.id = id    #id for objects of the same shape
        self.uid = -1   #unique id for the object, make a prime number
        self.name = name    #maybe remove? change to seperate objects?
        self.rotation = rotation
        self.moveable = moveable
        self.rotatable = rotatable

    def __str__(self):
        return f"{self.name} | Width: {self.width}, Depth: {self.depth}, Reserved Space: {self.reserved_space}, Rotation: {self.rotation}\n" \
                f"\tID: {self.id}, UID: {self.uid}"
    
    def shape_index(self):
        """
        Returns the shape index of the object for the moveable_shapes dictionary.
        """
        return f"{self.width}_{self.depth}_{self.reserved_space}_{self.rotatable}"

    def set_position(self, x: int, y: int, rotation: Rotation):
        """
        Set the position of the object.
        :param x: x position of the object.
        :param y: y position of the object.
        :param rotation: Rotation of the object.
        """
        self.x = x
        self.y = y
        self.rotation = rotation
    


class Room:
    """
    Class for the room. The room will have a width, height, name and a list of objects.
    Objects will be placed in the room and the space taken by the objects will be updated.
    """
    fobj = None # objective function
    X = [] # location in solution space

    def __init__(self, width: int, height: int, name: str = "Room"):
        """
        :param width: Width of the room.
        :param height: Height of the room.
        :param name: Name of the room.
        """
        self.width = width
        self.height = height
        self.name = name
        self.current_uid = 1 # must be prime numbers
        self.contiguous = None
        self.n_objects = 0

        self.uid_space = np.ones((width, height), dtype=int)   #unique id space for moving objects around
        self.taken_space = np.zeros((width, height), dtype=bool)#space taken by objects, including reserved space
        self.open_space = np.zeros((width, height), dtype=bool) #space that is not taken by objects, not including reserved space
        self.objects = defaultdict(Object) #objects in the room, key is the object uid
        self.moveable_shapes = defaultdict(list) #list of objects with the same shape, key is the shape index

        self.euc_norm = 0 #euclidean norm of the room, used for the objective function
        center = int(np.ceil(min(width, height)/2))
        for i in range(center):
            if width < 2 or height < 2:
                self.euc_norm += center*width*height
            else:
                self.euc_norm += 2*(width + height - 2)*(i + 1)
                width -= 2
                height -= 2
        

    def get_inventory(self) -> list[str]:
        """
        Returns the inventory of objects in the room.
        :return: List of objects in the room.
        """
        obj_list = [self.objects[uid].name for uid in self.objects.keys()]
        duplicates = Counter(obj_list)
        return [f"{k}: {v}" for k, v in duplicates.items()]


    def __str__(self):
        return f"Room | Width: {self.width}, Height: {self.height}, Number of Objects: {len(self.objects)} Contiguous: {self.contiguous}\n" \
                f"Objects: {self.get_inventory()}"


    def next_uid(self) -> int:
        """
        Returns the next prime number for unique id.
        """
        self.current_uid = nextprime(self.current_uid)
        return self.current_uid
    

    def add_object(self, object: Object, x: int, y: int, rotation: Rotation) -> bool:
        """
        Adds object to the room. If the object fits in the room, it will be added to the space.
        :param object: Object to add to the room.
        :param x: x position of the object.
        :param y: y position of the object.
        :param rotation: Rotation of the object.
        :return: True if object is added, False otherwise.
        """
        if self._fits(object, x, y, rotation):
            obj = Object(object.width, object.depth, object.reserved_space, object.name, x, y, object.id, rotation, object.moveable, object.rotatable)
            # Create new instance of object to avoid changing the original object between rooms
            obj.set_position(x, y, rotation)
            self._add_object_to_space(obj)
            return True
        else:
            return False
        

    def _fits(self, object: Object, x: int, y: int, rotation: Rotation) -> bool:
        """
        Checks if object fits in the room at position x, y with rotation.
        :param object: Object to check for fit.
        :param x: x position of the object.
        :param y: y position of the object.
        :param rotation: Rotation of the object.
        :return: True if object fits, False otherwise.
        """
        if(rotation == Rotation.TBD):
            return False
        if(x < 0 or y < 0 or y > self.height - 1 or x > self.width - 1): #check if object is outside of room
            return False
                
        depth = object.depth + object.reserved_space
        # Check direction of object, and if it fits in the room. Iterate through the space the object will take up.
        if (rotation == Rotation.UP) and (x + object.width <= self.width and y + depth <= self.height):
            for i in range(object.width):
                for j in range(depth):
                    if self.taken_space[x + i][y + j]:
                        return False
        elif (rotation == Rotation.RIGHT) and (x + depth <= self.width and y + object.width <= self.height):
            for i in range(depth):
                for j in range(object.width):
                    if self.taken_space[x + i][y + j]:
                        return False
        elif (rotation == Rotation.DOWN) and (x + object.width <= self.width and y + object.depth <= self.height) and (y - object.reserved_space >= 0):
            for i in range(object.width):
                for j in range(-object.reserved_space, object.depth):
                    if self.taken_space[x + i][y + j]:
                        return False
        elif (rotation == Rotation.LEFT) and (x + object.depth <= self.width and y + object.width <= self.height) and (x - object.reserved_space >= 0):
            for i in range(-object.reserved_space, object.depth):
                for j in range(object.width):
                    if self.taken_space[x + i][y + j]:
                        return False
        else:
            return False    # Doesn't fit in the room
        
        return True
        

    def _add_object_to_space(self, object: Object) -> None:
        """
        Updates the space taken by the object in the room.
        :param object: Object to add to the space.
        """
        self.n_objects += 1
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
        Returns the uid space for a plot.
        """
        return self.uid_space.transpose()
    

    def open_map(self) -> np.ndarray:
        """
        Returns the open space for a plot.
        """
        return self.open_space.transpose()
    

    def get_X(self) -> list[np.ndarray]:
        """
        Get solution space X.
        :return: Solution vector X.
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
    
    
    def set_X(self, X_float: list[np.ndarray]) -> bool:
        """
        Check for collisions in the new space.
        :param X_float: Solution vector holding new locations of objects
        :return: True if no collisions, False otherwise. Currently, always returns True.
        """
        X = [np.round(xi).astype(int) for xi in X_float]
        new_uid_space = np.ones((self.width, self.height), dtype=int) # May need to change the dtype to something larger
        new_open_space = np.zeros((self.width, self.height), dtype=bool)

        conflicts = []
        uid_to_X = defaultdict(tuple)

        if len(np.unique(self.uid_space)) != len(self.objects) + 1: # Check for missing objects
            raise AttributeError("MISSING OBJECTS")

        # Include the non-moveable objects. First objects should be moveable.
        for obj in self.objects.values():
            if obj.moveable:
                continue
            
            uid_to_X[obj.uid] = ()  # Add uid to the dictionary with no position in solution vector

            if obj.rotation == Rotation.UP:
                new_uid_space[obj.x : obj.x + obj.width, obj.y : obj.y + obj.depth + obj.reserved_space] *= obj.uid
                new_open_space[obj.x : obj.x + obj.width, obj.y : obj.y + obj.depth] = True
            elif obj.rotation == Rotation.RIGHT:
                new_uid_space[obj.x : obj.x + obj.depth + obj.reserved_space, obj.y : obj.y + obj.width] *= obj.uid
                new_open_space[obj.x : obj.x + obj.depth, obj.y : obj.y + obj.width] = True
            elif obj.rotation == Rotation.DOWN:
                new_uid_space[obj.x : obj.x + obj.width, obj.y - obj.reserved_space : obj.y + obj.depth] *= obj.uid
                new_open_space[obj.x : obj.x + obj.width, obj.y : obj.y + obj.depth] = True
            elif obj.rotation == Rotation.LEFT:
                new_uid_space[obj.x - obj.reserved_space : obj.x + obj.depth, obj.y : obj.y + obj.width] *= obj.uid
                new_open_space[obj.x : obj.x + obj.depth, obj.y : obj.y + obj.width] = True


        # Add in moveable objects to the new space
        for i, key in enumerate(self.moveable_shapes.keys()):
            for j, uid in enumerate(self.moveable_shapes[key]):
                obj = self.objects[uid]
                uid_to_X[uid] = (i,j)
                x, y = X[i][j][0], X[i][j][1]

                # If rotatable, update rotation
                if X[i][j].shape[0] == 3:
                    rotation = X[i][j][2] % 4
                    X[i][j][2] = rotation
                else:
                    rotation = obj.rotation


                if rotation == Rotation.UP:
                    # Check for out of bounds
                    if x < 0:
                        x = 0
                        X[i][j][0] = 0
                    elif x + obj.width > self.width:
                        x = self.width - obj.width
                        X[i][j][0] = self.width - obj.width
                    if y < 0:
                        y = 0
                        X[i][j][1] = 0
                    elif y + obj.depth + obj.reserved_space > self.height:
                        y = self.height - obj.depth - obj.reserved_space
                        X[i][j][1] = self.height - obj.depth - obj.reserved_space
                    # Check for conflicts
                    conf = np.where(new_uid_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] > 1) # Check for conflicts
                    if conf[0].size > 0:
                        uids = np.unique(new_uid_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space][conf])
                        conflicts.append(np.concatenate(([uid],uids)))
                    # Add object to the space
                    new_uid_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] *= obj.uid
                    new_open_space[x : x + obj.width, y : y + obj.depth] = True

                elif rotation == Rotation.RIGHT:
                    # Check for out of bounds
                    if x < 0:
                        x = 0
                        X[i][j][0] = 0
                    elif x + obj.depth + obj.reserved_space > self.width:
                        x = self.width - obj.depth - obj.reserved_space
                        X[i][j][0] = self.width - obj.depth - obj.reserved_space
                    if y < 0:   
                        y = 0
                        X[i][j][1] = 0
                    elif y + obj.width > self.height:
                        y = self.height - obj.width
                        X[i][j][1] = self.height - obj.width
                    # Check for conflicts
                    conf = np.where(new_uid_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] > 1) # Check for conflicts
                    if conf[0].size > 0:
                        uids = np.unique(new_uid_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width][conf])
                        conflicts.append(np.concatenate(([uid],uids)))
                    # Add object to the space
                    new_uid_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] *= obj.uid
                    new_open_space[x : x + obj.depth, y : y + obj.width] = True

                elif rotation == Rotation.DOWN:
                    # Check for out of bounds
                    if x < 0:
                        x = 0
                        X[i][j][0] = 0
                    elif x + obj.width > self.width:
                        x = self.width - obj.width
                        X[i][j][0] = self.width - obj.width
                    if y < obj.reserved_space:
                        y = obj.reserved_space
                        X[i][j][1] = obj.reserved_space
                    elif y + obj.depth > self.height:
                        y = self.height - obj.depth
                        X[i][j][1] = self.height - obj.depth
                    # Check for conflicts
                    conf = np.where(new_uid_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] > 1) # Check for conflicts
                    if conf[0].size > 0:
                        uids = np.unique(new_uid_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth][conf])
                        conflicts.append(np.concatenate(([uid],uids)))
                    # Add object to the space
                    new_uid_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] *= obj.uid
                    new_open_space[x : x + obj.width, y : y + obj.depth] = True

                elif rotation == Rotation.LEFT:
                    # Check for out of bounds
                    if x < obj.reserved_space:
                        x = obj.reserved_space
                        X[i][j][0] = obj.reserved_space
                    elif x + obj.depth > self.width:
                        x = self.width - obj.depth
                        X[i][j][0] = self.width - obj.depth
                    if y < 0:
                        y = 0
                        X[i][j][1] = 0
                    elif y + obj.width > self.height:
                        y = self.height - obj.width
                        X[i][j][1] = self.height - obj.width
                    # Check for conflicts
                    conf = np.where(new_uid_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] > 1) # Check for conflicts
                    if conf[0].size > 0:
                        uids = np.unique(new_uid_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width][conf])
                        conflicts.append(np.concatenate(([uid],uids)))
                    # Add object to the space
                    new_uid_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] *= obj.uid
                    new_open_space[x : x + obj.depth, y : y + obj.width] = True

                else:
                    raise AttributeError("UNKNOWN")
 

        if len(conflicts) > 0:
            # Update conflict list
            conflicts = list(np.unique([u for conflict in conflicts for u in conflict]))
            conflicts = [uid for uid in conflicts if uid in uid_to_X.keys()]
            unmoved = []
            # Remove conflicts from the space
            for uid in conflicts:
                if uid_to_X[uid] == (): # Filter out objects that cannot move
                    unmoved.append(uid) 
                    continue
                
                i,j = uid_to_X[uid]
                if(X[i][j] == self.X[i][j]).all():  # If the object hasn't moved, give it the position
                    unmoved.append(uid) 
                else: # If the object has moved, remove from space
                    x, y = X[i][j][0], X[i][j][1]
                    obj = self.objects[uid]
                    rotation = X[i][j][2] if X[i][j].shape[0] == 3 else obj.rotation
                    # Get position of the object and remove from space
                    if rotation == Rotation.UP:
                        x0 , xf, y0, yf = x, x + obj.width, y, y + obj.depth + obj.reserved_space
                        new_open_space[x : x + obj.width, y : y + obj.depth] = False
                    elif rotation == Rotation.RIGHT:
                        x0 , xf, y0, yf = x, x + obj.depth + obj.reserved_space, y, y + obj.width
                        new_open_space[x : x + obj.depth, y : y + obj.width] = False
                    elif rotation == Rotation.DOWN:
                        x0 , xf, y0, yf = x, x + obj.width, y - obj.reserved_space, y + obj.depth
                        new_open_space[x : x + obj.width, y : y + obj.depth] = False
                    elif rotation == Rotation.LEFT:
                        x0 , xf, y0, yf = x - obj.reserved_space, x + obj.depth, y, y + obj.width
                        new_open_space[x : x + obj.depth, y : y + obj.width] = False
                    else:
                        raise AttributeError("UNKNOWN")
                    # Remove the object from the space
                    new_uid_space[x0 : xf, y0 : yf] = (new_uid_space[x0 : xf, y0 : yf]/uid).astype(int)


            # Add objects that haven't moved back into the open space
            for uid in unmoved:
                conflicts.remove(uid)
                obj = self.objects[uid]
                if obj.rotation == Rotation.UP:
                    new_open_space[obj.x : obj.x + obj.width, obj.y : obj.y + obj.depth] = True
                elif obj.rotation == Rotation.RIGHT:
                    new_open_space[obj.x : obj.x + obj.depth, obj.y : obj.y + obj.width] = True
                elif obj.rotation == Rotation.DOWN:
                    new_open_space[obj.x : obj.x + obj.width, obj.y : obj.y + obj.depth] = True
                elif obj.rotation == Rotation.LEFT:
                    new_open_space[obj.x : obj.x + obj.depth, obj.y : obj.y + obj.width] = True
                else:
                    raise AttributeError("UNKNOWN")
                
            # Resolve conflicts with noise separation
            self._noise_separation(X, new_uid_space, new_open_space, uid_to_X, conflicts)


        if False: # Ignore this for now
            # Check if the new space is contiguous
            labels, num_labels = label(new_open_space, connectivity=1, background=1, return_num=True)
            if num_labels > 1:
                raise NotImplementedError("Space is not contiguous and needs to be resolved.")
                #return False # Change to fix the space
        else:   
            self.contiguous = self._is_contiguous(new_open_space)    # Check if the space is contiguous
    

        if len(np.unique(new_uid_space)) != len(self.objects) + 1: # Check for missing objects
            raise AttributeError("MISSING OBJECTS")

        # Update to the new space
        self.uid_space = new_uid_space
        self.open_space = new_open_space
        for uid in uid_to_X.keys():
            if uid_to_X[uid] == ():
                continue
            i, j = uid_to_X[uid]
            if self.X[i].shape[1] == 3:
                self.objects[uid].set_position(X[i][j][0], X[i][j][1], X[i][j][2])
            else:
                self.objects[uid].set_position(X[i][j][0], X[i][j][1], self.objects[uid].rotation)
        return True


    def _noise_separation(self, X: list[np.ndarray], new_uid_space: np.ndarray, new_open_space: np.ndarray, uid_to_X: dict, conflicts: list[int]) -> None:
        """
        Resolve conflicts with noise separation. This will incrimentally 
        add noise to the position of the objects until conflicts are resolved.
        :param X: Solution vector holding new locations of objects
        :param new_uid_space: 
        """
        sigma = 1 # Noise parameter that will grow with iterations

        # Resolve conflicts with noise separation
        while len(conflicts) > 0:
            resolve_space = np.copy(new_uid_space)
            resolve_locations = defaultdict(tuple)
            confs = []
            for uid in conflicts:
                if uid_to_X[uid] == (): # If object cannot move, skip
                    continue

                i, j = uid_to_X[uid]
                # Add noise to the position
                x, y = int(X[i][j][0] + sigma*np.random.randn(1)[0]), int(X[i][j][1] + sigma*np.random.randn(1)[0])

                obj = self.objects[uid]
                rotation = X[i][j][2] if X[i][j].shape[0] == 3 else obj.rotation
                # Update the space with the new position
                if rotation == Rotation.UP:
                    if x < 0:
                        x = 0
                    elif x + obj.width > self.width:
                        x = self.width - obj.width
                    if y < 0:
                        y = 0
                    elif y + obj.depth + obj.reserved_space > self.height:
                        y = self.height - obj.depth - obj.reserved_space

                    conf = np.where(resolve_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] > 1)
                    if conf[0].size > 0:
                        uids = np.unique(resolve_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space][conf])
                        confs.append(np.concatenate(([uid],uids)))
                    resolve_locations[uid] = (x, y)
                    resolve_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] *= obj.uid

                elif rotation == Rotation.RIGHT:
                    if x < 0:
                        x = 0
                    elif x + obj.depth + obj.reserved_space > self.width:
                        x = self.width - obj.depth - obj.reserved_space
                    if y < 0:
                        y = 0
                    elif y + obj.width > self.height:
                        y = self.height - obj.width

                    conf = np.where(resolve_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] > 1)
                    if conf[0].size > 0:
                        uids = np.unique(resolve_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width][conf])
                        confs.append(np.concatenate(([uid],uids)))
                    resolve_locations[uid] = (x, y)
                    resolve_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] *= obj.uid

                elif rotation == Rotation.DOWN:
                    if x < 0:
                        x = 0
                    elif x + obj.width > self.width:
                        x = self.width - obj.width
                    if y < obj.reserved_space:
                        y = obj.reserved_space
                    elif y + obj.depth > self.height:
                        y = self.height - obj.depth

                    conf = np.where(resolve_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] > 1)
                    if conf[0].size > 0:
                        uids = np.unique(resolve_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth][conf])
                        confs.append(np.concatenate(([uid],uids)))
                    resolve_locations[uid] = (x, y)
                    resolve_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] *= obj.uid

                elif rotation == Rotation.LEFT:
                    if x < obj.reserved_space:
                        x = obj.reserved_space
                    elif x + obj.depth > self.width:
                        x = self.width - obj.depth
                    if y < 0:
                        y = 0
                    elif y + obj.width > self.height:
                        y = self.height - obj.width

                    conf = np.where(resolve_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] > 1)
                    if conf[0].size > 0:
                        uids = np.unique(resolve_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width][conf])
                        confs.append(np.concatenate(([uid],uids)))
                    resolve_locations[uid] = (x, y)
                    resolve_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] *= obj.uid

                else:
                    raise AttributeError("UNKNOWN")
            
            # Get new conflict list
            new_conflicts = list(np.unique([u for conflict in confs for u in conflict]))
            new_conflicts = [uid for uid in new_conflicts if uid in uid_to_X.keys()] # Remove uids that are not in the solution space
            # Update conflicts with intersection of old and new conflicts to get unresolved conflicts
            new_conflicts = set(conflicts).intersection(set(new_conflicts))
            # Add resolved locations to the space
            resolved_conflicts = set(conflicts).difference(new_conflicts)
            for uid in resolved_conflicts:
                x, y = resolve_locations[uid]
                i, j = uid_to_X[uid]
                X[i][j][0], X[i][j][1] = x, y
                obj = self.objects[uid]
                rotation = X[i][j][2] if X[i][j].shape[0] == 3 else obj.rotation
                if rotation == Rotation.UP:
                    new_uid_space[x : x + obj.width, y : y + obj.depth + obj.reserved_space] = obj.uid
                    new_open_space[x : x + obj.width, y : y + obj.depth] = True
                elif rotation == Rotation.RIGHT:
                    new_uid_space[x : x + obj.depth + obj.reserved_space, y : y + obj.width] = obj.uid
                    new_open_space[x : x + obj.depth, y : y + obj.width] = True
                elif rotation == Rotation.DOWN:
                    new_uid_space[x : x + obj.width, y - obj.reserved_space : y + obj.depth] = obj.uid
                    new_open_space[x : x + obj.width, y : y + obj.depth] = True
                elif rotation == Rotation.LEFT:
                    new_uid_space[x - obj.reserved_space : x + obj.depth, y : y + obj.width] = obj.uid
                    new_open_space[x : x + obj.depth, y : y + obj.width] = True
                else:
                    raise AttributeError("UNKNOWN")

            conflicts = list(new_conflicts)
            sigma += 1 # Increase noise parameter
            if sigma > max(self.width, self.height):
                sigma = 1


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
    

    def _is_contiguous(self, space: np.ndarray) -> bool:
        """
        Check if the space is contiguous.
        :param space: boolean np.ndarray
        :return: True if contiguous, False otherwise
        """
        _, num = label(space, connectivity=1, background=1, return_num=True)
        return num == 1


    def evaluate(self, optimize_type: str = "taxi_cab_dist") -> float:
        """
        Evaluate the objective function.
        :param optimize_type: Type of optimization. Default is "taxi_cab_dist".
        :return: Evaluation of the objective function.
        """

        if optimize_type == "open_dist" or optimize_type == "open_distance":
            contour_map = np.zeros((self.width, self.height), dtype=float)
            for i in range(self.width):
                for j in range(self.height):
                    open_width = self._get_open_width(i,j)
                    open_height = self._get_open_height(i,j)
                    contour_map[i][j] = open_width*open_height
            eval = np.sum(contour_map)/(np.prod(contour_map.shape)**2)

        elif optimize_type == "taxi_cab_distance" or optimize_type == "taxi_cab_dist":
            contour_map = (~self.open_space).astype(float)
            contour_map[1:self.width-1,1:self.height-1] *= self.height # Could be height or width, just large value
            nodes = np.transpose(np.where(contour_map == 0))
            largest = 1
            
            while(len(nodes) > 0):
                for node in nodes:
                    for neighbor in [(node[0]+1, node[1]), (node[0]-1, node[1]), (node[0], node[1]+1), (node[0], node[1]-1)]: # Check neighbors
                        if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height and contour_map[neighbor] > contour_map[node[0], node[1]]:
                            contour_map[neighbor] = largest
                nodes = np.transpose(np.where(contour_map == largest))
                largest += 1
                
            eval = np.sum(contour_map)/(np.prod(contour_map.shape)**2)

        elif optimize_type == "euclidean_distance" or optimize_type == "euclidean_dist":
            raise NotImplementedError("Not implemented yet.")

        self.fobj = eval
        return eval


    def _get_open_width(self, x: int, y: int) -> int:
        """
        Get the width of the open space at position x, y.
        :param x: x position
        :param y: y position
        :return: width of open space at position x, y
        """
        width = 0
        for i in range(x, self.width):
            if not self.open_space[i][y]:
                width += 1
            else:
                break
        for i in range(x, -1, -1):
            if not self.open_space[i][y]:
                width += 1
            else:
                break
        return width
    
    
    def _get_open_height(self, x: int, y: int) -> int:
        """
        Get the height of the open space at position x, y.
        :param x: x position
        :param y: y position
        :return: height of open space at position x, y
        """
        height = 0
        for j in range(y, self.height):
            if not self.open_space[x][j]:
                height += 1
            else:
                break
        for j in range(y, -1, -1):
            if not self.open_space[x][j]:
                height += 1
            else:
                break
        return height
    




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

    print(room)
    print("Objective function value: ",room.evaluate("taxi_cab_dist"))
    print(f"Solution: {room.get_X()}")

    plt.subplot(1, 2, 1)
    plt.imshow(room.uid_map(), origin='lower')
    plt.title("UID Space")

    plt.subplot(1, 2, 2)
    plt.imshow(room.open_map(), origin='lower')
    plt.title("Open Space")

    plt.show()