import numpy as np
from Storage import Object, Room, Rotation
from random import randint, random


class Firefly:
    """
    This class holds the attributes of a firefly in the Firefly Algorithm (FA).
    """
    def __init__(self, objects:list[Object], width:int, height:int, name:str="Room"):
        """
        Initializes the firefly to hold a solution vector for the room layout.
        """
        self.Room = Room(width, height, name)
        self.generate_random_solution(objects)

    def generate_random_solution(self, objects:list[Object]) -> None:
        """
        Generates a random solution vector for the room layout.
        """
        # Add objects to the room in random positions
        for obj in objects:
            valid = False
            while not valid:
                if obj.rotatable:
                    rotation = randint(0, 3)
                    if rotation == Rotation.UP:
                        x = randint(0, self.Room.width - obj.width)
                        y = randint(0, self.Room.height - obj.depth - obj.reserved_space)
                    elif rotation == Rotation.RIGHT:
                        x = randint(0, self.Room.width - obj.depth - obj.reserved_space)
                        y = randint(0, self.Room.height - obj.width)
                    elif rotation == Rotation.DOWN:
                        x = randint(0, self.Room.width - obj.width)
                        y = randint(-obj.reserved_space, self.Room.height - obj.depth)
                    elif rotation == Rotation.LEFT:
                        x = randint(-obj.reserved_space, self.Room.width - obj.depth)
                        y = randint(0, self.Room.height - obj.width)
                    else:
                        raise ValueError("Invalid rotation value")
                    valid =  self.Room.add_object(obj, x, y, rotation)
                else:
                    valid =  self.Room.add_object(obj, x, y, obj.rotation)

    

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
        # Maybe add in 
        # :param fobj: Objective function to be minimized
        # :param lb: Lower bounds of the search space
        # :param ub: Upper bounds of the search space
        #self.objects = objects
        #self.width = width
        #self.height = height
        self.N = N
        self.T = T
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.fireflies = [Firefly(objects, width, height, name + str(i)) for i in range(self.N)]

    def calc_distance(self, firefly1:Firefly, firefly2:Firefly) -> list[np.ndarray]:
        """
        Calculates the Euclidean distance between two fireflies.
        :param room1: First firefly
        :param room2: Second firefly
        """
        X1 = firefly1.Room.get_X()
        X2 = firefly2.Room.get_X()
        r = []
        for xi, xj in zip(X1, X2):
            r.append(np.sqrt((xi - xj)**2))
        return r
    
    def move_firefly(self, firefly1:Firefly, firefly2:Firefly) -> float:
        """
        Move firefly1 towards firefly2.
        :param firefly1: First firefly
        :param firefly2: Second firefly
        """
        X = []
        X1 = firefly1.Room.get_X()
        X2 = firefly2.Room.get_X()
        for xi1, xi2 in zip(X1, X2):
            r = np.sqrt((xi1 - xi2)**2)
            beta = self.beta0 * np.exp(-self.gamma * r**2) 
            xi = xi1 + beta * (xi2 - xi1) + self.alpha * (random() - 0.5)
            X.append(xi)
        firefly1.Room.set_X(X)

    def optimize(self) -> Firefly:
        """
        Optimizes the room layout using the Firefly Algorithm (FA).
        """
        # Evaluate the objective function for each firefly
        for firefly in self.fireflies:
            firefly.Room.evaluate()
        # Sort fireflies by objective function value
        self.fireflies.sort(key=lambda x: x.Room.fobj)
        # Initialize the best solution
        best = self.fireflies[0]
        # Main loop
        for t in range(self.T):
            # Update fireflies
            for i in range(self.N):
                for j in range(self.N):
                    if self.fireflies[i].Room.fobj < self.fireflies[j].Room.fobj:
                        # Calculate the attractiveness
                        self.move_firefly(self.fireflies[i], self.fireflies[j])
                                #beta = self.beta0 * np.exp(-self.gamma * r**2)
                        # Update the position of the firefly

                                #fireflies[i].Room.x = fireflies[i].Room.x * (1 - beta) + fireflies[j].Room.x * beta + self.alpha * (random() - 0.5)
                                #fireflies[i].Room.y = fireflies[i].Room.y * (1 - beta) + fireflies[j].Room.y * beta + self.alpha * (random() - 0.5)
                                
                        # Evaluate the objective function
                        self.fireflies[i].Room.evaluate()
            # Sort fireflies by objective function value
            self.fireflies.sort(key=lambda x: x.Room.fobj)
            # Update the best solution
            if self.fireflies[0].Room.fobj < best.Room.fobj:
                best = self.fireflies[0]
        return best
