import numpy as np
from Storage import Object, Room
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
        self.generate_random_solution()

    def generate_random_solution(self) -> None:
        """
        Generates a random solution vector for the room layout.
        """
        # Add objects to the room in random positions
        for obj in self.objects:
            valid = False
            while not valid:
                x = randint(0, self.width - obj.width)
                y = randint(0, self.height - obj.height)
                if obj.rotatable:
                    rotation = randint(0, 3)
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
        self.objects = objects
        self.width = width
        self.height = height
        self.N = N
        self.T = T
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma

    def optimize(self) -> Firefly:
        """
        Optimizes the room layout using the Firefly Algorithm (FA).
        """
        # Initialize fireflies
        fireflies = [Firefly(self.objects, self.width, self.height, "Room" + str(i)) for i in range(self.N)]
        # Evaluate the objective function for each firefly
        for firefly in fireflies:
            firefly.Room.evaluate()
        # Sort fireflies by objective function value
        fireflies.sort(key=lambda x: x.Room.fobj)
        # Initialize the best solution
        best = fireflies[0]
        # Main loop
        for t in range(self.T):
            # Update fireflies
            for i in range(self.N):
                for j in range(self.N):
                    if fireflies[i].Room.fobj < fireflies[j].Room.fobj:
                        # Calculate the Euclidean distance between fireflies
                        r = np.sqrt((fireflies[i].Room.x - fireflies[j].Room.x)**2 + (fireflies[i].Room.y - fireflies[j].Room.y)**2)
                        # Calculate the attractiveness
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        # Update the position of the firefly

                        #fireflies[i].Room.x = fireflies[i].Room.x * (1 - beta) + fireflies[j].Room.x * beta + self.alpha * (random() - 0.5)
                        #fireflies[i].Room.y = fireflies[i].Room.y * (1 - beta) + fireflies[j].Room.y * beta + self.alpha * (random() - 0.5)
                        
                        # Evaluate the objective function
                        fireflies[i].Room.evaluate()
            # Sort fireflies by objective function value
            fireflies.sort(key=lambda x: x.Room.fobj)
            # Update the best solution
            if fireflies[0].Room.fobj < best.Room.fobj:
                best = fireflies[0]
        return best
