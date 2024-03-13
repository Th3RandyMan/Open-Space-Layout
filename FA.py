import numpy as np
from Storage import Object, Room, Rotation
from random import randint, random


class Firefly:
    """
    This class holds the attributes of a firefly in the Firefly Algorithm (FA).
    """
    fobj = None # Objective function value
    def __init__(self, objects:list[Object], width:int, height:int, name:str="Room"):
        """
        Initializes the firefly to hold a solution vector for the room layout.
        """
        self.room = Room(width, height, name)
        self.generate_random_solution(objects)

    def generate_random_solution(self, objects:list[Object]) -> None:
        """
        Generates a random solution vector for the room layout.
        """
        # Add objects to the room in random positions
        for id, obj in enumerate(objects):
            obj.id = id # Set the id of the object by its index in the list
            valid = False
            # Try to add the object to the room until a valid position is found
            while not valid:
                # Generate random rotation if the object is rotatable
                if obj.rotatable:
                    obj.rotation = randint(0, 3)
                
                # Generate random position
                if obj.rotation == Rotation.UP:
                    x = randint(0, self.room.width - obj.width)
                    y = randint(0, self.room.height - obj.depth - obj.reserved_space)
                elif obj.rotation == Rotation.RIGHT:
                    x = randint(0, self.room.width - obj.depth - obj.reserved_space)
                    y = randint(0, self.room.height - obj.width)
                elif obj.rotation == Rotation.DOWN:
                    x = randint(0, self.room.width - obj.width)
                    y = randint(-obj.reserved_space, self.room.height - obj.depth)
                elif obj.rotation == Rotation.LEFT:
                    x = randint(-obj.reserved_space, self.room.width - obj.depth)
                    y = randint(0, self.room.height - obj.width)
                else:
                    raise ValueError("Invalid rotation value")
                # Add object to the room if the position is valid
                valid =  self.room.add_object(obj, x, y, obj.rotation)

    def __str__(self) -> str:
        """
        Returns the string representation of the firefly.
        """
        return str(self.room)
    
    def get_X(self) -> list[np.ndarray]:
        """
        Returns the solution vector of the firefly.
        """
        return self.room.get_X()
    
    def set_X(self, X:list[np.ndarray]) -> None:
        """
        Sets the solution vector of the firefly.
        """
        self.room.set_X(X)

    def evaluate(self) -> float:
        """
        Evaluates the objective function of the firefly.
        """
        self.room.evaluate()
        self.fobj = self.room.fobj
        return self.fobj
    

class FA:
    """
    This class implements the Firefly Algorithm (FA) for optimization.
    """

    def __init__(self, objects:list[Object], width:int, height:int, N:int, T:int, alpha:float=0.2, beta0:float=1.0, gamma:float=1.0, name:str="Room"):
        """
        Initializes the Firefly Algorithm (FA) with the following parameters:
        :param objects: List of objects to be placed in the room
        :param width: Width of the room
        :param height: Height of the room
        :param N: Number of fireflies
        :param T: Number of iterations
        :param alpha: Randomness parameter
        :param beta0: Attraction coefficient
        :param gamma: Light absorption coefficient
        """
        self.N = N
        self.T = T
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.fireflies = [Firefly(objects, width, height, name + str(i)) for i in range(self.N)]
    
    def move_firefly(self, firefly1:Firefly, firefly2:Firefly) -> float:
        """
        Move firefly1 towards firefly2.
        :param firefly1: First firefly
        :param firefly2: Second firefly
        """
        X = []
        X1 = firefly1.get_X()
        X2 = firefly2.get_X()
        for xi1, xi2 in zip(X1, X2):
            # Calculate the Euclidean distance
            r = np.sqrt((xi1 - xi2)**2)
            # Calculate the attractiveness
            beta = self.beta0 * np.exp(-self.gamma * r**2) 
            # Update the position of firefly1
            xi = xi1 + beta * (xi2 - xi1) + self.alpha * (random() - 0.5)
            X.append(xi)
        firefly1.set_X(X)

    def optimize(self) -> Room:
        """
        Optimizes the room layout using the Firefly Algorithm (FA).
        """
        # Evaluate the objective function for each firefly
        for firefly in self.fireflies:
            firefly.evaluate()
        # Sort fireflies by objective function value
        self.fireflies.sort(key=lambda x: x.fobj)
        # Initialize the best solution
        best = self.fireflies[0]
        # Main loop
        for t in range(self.T):
            # Update fireflies
            for i in range(self.N):
                for j in range(self.N):
                    if self.fireflies[i].fobj < self.fireflies[j].fobj:
                        # Move firefly1 towards firefly2
                        self.move_firefly(self.fireflies[i], self.fireflies[j])     
                        # Evaluate the objective function
                        self.fireflies[i].evaluate()
            # Sort fireflies by objective function value
            self.fireflies.sort(key=lambda x: x.fobj)
            # Update the best solution
            if self.fireflies[0].fobj < best.fobj:
                best = self.fireflies[0]
        return best.room


if __name__ == "__main__":
    # Example usage of the Firefly Algorithm (FA)
    table1 = Object(10, 10, 5, 1, "Table") 
    couch1 = Object(30, 10, 8, 2, "Couch")
    door1 = Object(10, 0, 8, 3, "Door", rotation=Rotation.UP, rotatable=False, moveable=False)

    width = 100
    height = 100
    objects = [table1, couch1, door1]
    N = 10  # Number of fireflies
    T = 10  # Number of iterations

    FA = FA(objects, width, height, N, T)
    room = FA.optimize()

    print(room)
    print(f"Solution: {room.get_X()}")

    from matplotlib import pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(room.uid_map(), origin='lower')
    plt.title("UID Space")

    plt.subplot(1, 2, 2)
    plt.imshow(room.open_map(), origin='lower')
    plt.title("Open Space")

    plt.show()