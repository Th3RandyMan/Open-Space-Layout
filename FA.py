import numpy as np
from Storage import Object, Room, Rotation
from random import randint, random
from matplotlib import pyplot as plt
from tqdm import tqdm
#import signal


# def timeout_handler(signum, frame):
#     raise Exception("Timeout when creating random room layout.")

class Firefly:
    """
    This class holds the attributes of a firefly in the Firefly Algorithm (FA).
    """
    fobj = None # Objective function value
    def __init__(self, objects:list[Object], width:int, height:int, name:str="Room", id:int=-1, timeout:int=10):
        """
        Initializes the firefly to hold a solution vector for the room layout.
        :param objects: List of objects to be placed in the room.
        :param width: Width of the room.
        :param height: Height of the room.
        :param name: Name of the room.
        :param id: ID of the firefly.
        :param timeout: Timeout for generating random room layout. Currently not implemented.
        """
        self.id = id
        self.room = Room(width, height, name)
        self.generate_random_solution(objects, timeout)
        self.fobj = None  # Objective function value
        self.moved = False  # Flag to check if the firefly moved

    def generate_random_solution(self, objects:list[Object], timeout: int=10) -> None:
        """
        Generates a random solution for the room layout.
        :param objects: List of objects to be placed in the room.
        :param timeout: Timeout for generating random room layout. Currently not implemented.
        """
        # Find unmovable objects and remove them from the list
        unmovable = [obj for obj in objects if not obj.moveable]
        for id, obj in enumerate(unmovable):
            if not self.room.add_object(obj, obj.x, obj.y, obj.rotation):
                raise ValueError("Invalid position")
            obj.id = id # Set the id of the object by its index in the list
        id_offset = len(unmovable)

        # Create timeout for generating random room layout
        # signal.signal(signal.SIGINT, timeout_handler)
        # signal.alarm(timeout)  # Number of seconds before timeout

        # Add objects to the room in random positions
        for id, obj in enumerate(objects):
            if not obj.moveable:
                continue
            
            obj.id = id + id_offset # Set the id of the object by its index in the list
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
                valid = self.room.add_object(obj, x, y, obj.rotation)

        # Cancel the timeout
        #signal.alarm(0)

    def __str__(self) -> str:
        """
        Returns the string representation of the firefly.
        """
        return str(self.room)
    
    def get_X(self) -> list[np.ndarray]:
        """
        Returns the solution vector of the firefly.
        :return: Solution vector of the firefly.
        """
        return self.room.get_X()
    
    def set_X(self, X:list[np.ndarray]) -> bool:
        """
        Sets the solution vector of the firefly.
        :param X: Solution vector of the firefly.
        :return: True if the solution vector is valid, False otherwise.
        """
        return self.room.set_X(X)

    def evaluate(self, optimize_type) -> float:
        """
        Evaluates the objective function of the firefly.
        :param optimize_type: Type of evaluation function to use. Options are "taxi_cab_dist" or "open_dist".
        :return: Objective function value of the firefly.
        """
        self.fobj = self.room.evaluate(optimize_type)
        return self.fobj
    

class FA:
    """
    This class implements the Firefly Algorithm (FA) for optimization.
    """

    def __init__(self, objects:list[Object], width:int, height:int, N:int, T:int, alpha:tuple[float,float]=(2,0.2), beta0:tuple[float,float]=(5,0.5), gamma:float=0.01, name:str="Room"):
        """
        Initializes the Firefly Algorithm (FA) with the following parameters:
        :param objects: List of objects to be placed in the room
        :param width: Width of the room
        :param height: Height of the room
        :param N: Number of fireflies
        :param T: Number of iterations
        :param alpha: Randomness parameter. The first value is for position and the second value is for rotation
        :param beta0: Attraction coefficient. The first value is for position and the second value is for rotation
        :param gamma: Light absorption coefficient.
        """
        self.N = N
        self.T = T
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.fireflies = [Firefly(objects, width, height, name + str(i), i) for i in range(self.N)]
        self.best = None
    
    def move_firefly(self, firefly1:Firefly, firefly2:Firefly) -> None:
        """
        Move firefly1 towards firefly2.
        :param firefly1: First firefly
        :param firefly2: Second firefly
        """
        X = []
        X1 = firefly1.get_X()
        X2 = firefly2.get_X()
        for xi1, xi2 in zip(X1, X2):
            # Fix rotations if object is rotatable
            if xi1.shape[1] == 3:
                loc = np.intersect1d(np.where(xi1[:, 2] == 0), np.where(xi2[:, 2] == 3))
                xi1[loc, 2] = 4
                loc = np.intersect1d(np.where(xi1[:, 2] == 3), np.where(xi2[:, 2] == 0))
                xi2[loc, 2] = 4
                
            # Calculate the Euclidean distance
            r = np.sqrt((xi1 - xi2)**2)
            # Calculate the attractiveness
            beta = self.beta0[0] * np.exp(-self.gamma * r**2) 
            # Update the position of firefly1
            noise = np.random.randn(*xi2.shape)
            noise[:, 0:2] = self.alpha[0] * noise[:, 0:2]    # Maybe adjust parameter to be different for rotation
            
            if xi1.shape[1] == 3: # If rotatable
                beta[:,2] = self.beta0[1] * np.exp(-self.gamma * r[:,2]**2) 
                noise[:, 2] = self.alpha[1] * noise[:, 2]

            xi = xi1 + beta * (xi2 - xi1) + noise   
            X.append(xi)
        valid = firefly1.set_X(X)
        if not valid:
            raise ValueError("Invalid position") # Should avoid this case
        else:
            moved = []
            for xi1, xi in zip(X1, X):
                moved.append(np.array_equal(xi1, xi.astype(int)))
            firefly1.moved = moved.count(True) != len(moved)    # Check if the firefly moved

    def optimize(self, optimize_type = "taxi_cab_dist") -> Room:
        """
        Optimizes the room layout using the Firefly Algorithm (FA).
        :param optimize_type: Type of evaluation function to use. Options are "taxi_cab_dist" or "open_dist".
        :return: Best room layout found
        """
        # Evaluate the objective function for each firefly
        for firefly in self.fireflies:
            firefly.evaluate(optimize_type)
        # Sort fireflies by objective function value
        self.fireflies.sort(key=lambda x: x.fobj, reverse=True)
        # Initialize the best solution
        best = self.fireflies[0]
        updates = [(0, best.fobj)]
        # Main loop
        with tqdm(total=self.T*self.N) as pbar:
            for t in range(self.T):
                moved = []
                # Update fireflies
                for i in range(self.N):
                    for j in range(self.N):
                        if self.fireflies[i].fobj < self.fireflies[j].fobj:
                            # Move firefly1 towards firefly2
                            self.move_firefly(self.fireflies[i], self.fireflies[j])  
                            moved.append(self.fireflies[i].moved)
                            # Evaluate the objective function
                            self.fireflies[i].evaluate(optimize_type)
                    pbar.update(1)
                # Sort fireflies by objective function value
                self.fireflies.sort(key=lambda x: x.fobj, reverse=True)
                # Update the best solution
                if self.fireflies[0].fobj > best.fobj:
                    updates.append((t+1, best.fobj))
                    best = self.fireflies[0]
                if moved.count(True) == 0:
                    print(f"Converged at iteration {t}")
                    break
        
        for t, fobj in updates:
            print(f"{t}/{self.T} | {fobj}")

        self.best = best
        return best.room



class DbFA:
    """
    This class implements the Distance-based Firefly Algorithm (DbFA) for optimization.
    """

    def __init__(self, objects:list[Object], width:int, height:int, N:int, T:int, ohm:tuple[float,float]=(2,0.2), beta0:tuple[float,float]=(5,0.5), lamb:float=0.01, name:str="Room"):
        """
        Initializes the Firefly Algorithm (FA) with the following parameters:
        :param objects: List of objects to be placed in the room
        :param width: Width of the room
        :param height: Height of the room
        :param N: Number of fireflies
        :param T: Number of iterations
        :param ohm: Randomness parameter ohmega
        :param beta0: Attraction coefficient
        :param lamb: Light absorption parameter lambda
        """
        self.N = N
        self.T = T
        self.ohm = ohm
        self.beta0 = beta0
        self.lamb = lamb
        self.fireflies = [Firefly(objects, width, height, name + str(i), i) for i in range(self.N)]
        #self.sg = np.zeros((self.N, self.N, 2)) # Known distances between fireflies, 0 for position, 1 for rotation


    def alpha(self, t:int, state:int) -> float:
        """
        Function to calculate the randomness parameter alpha.
        :param t: Current iteration
        :param state: State of the object (0 for position, 1 for rotation)
        :return: Randomness parameter alpha
        """
        return (0.5^(t/self.T))*np.exp(-self.ohm[state]*(self.T - t)/self.T)
    

    def gamma(self, t:int, state:int) -> float:
        """
        Function to calculate the light absorption parameter gamma.
        :param t: Current iteration
        :param state: State of the object (0 for position, 1 for rotation)
        :return: Light absorption parameter gamma
        """
        raise NotImplementedError("This function is not implemented yet.")
        f = (sg - smin)/(smax - smin)
        return 1/(1 + (10E3)*self.lamb[state]*np.exp(-self.lamb[state]*f))


    def _update_sg(self, r:np.ndarray, fid1:int, fid2:int) -> None:
        """
        Update known distances between fireflies.
        :param r: Distance between fireflies
        :param fid1: ID of the first firefly
        :param fid2: ID of the second firefly
        """
        raise NotImplementedError("This function is not implemented yet.")
        # This would not work since r will vary depending on the objects.
        self.sg[fid1, fid2] = r
        self.sg[fid2, fid1] = r

    
    def move_firefly(self, firefly1:Firefly, firefly2:Firefly, t:int) -> None:
        """
        Move firefly1 towards firefly2.
        :param firefly1: First firefly
        :param firefly2: Second firefly
        :param t: Current iteration
        """
        X = []
        X1 = firefly1.get_X()
        X2 = firefly2.get_X()
        for xi1, xi2 in zip(X1, X2):
            # Fix rotations if object is rotatable
            if xi1.shape[1] == 3:
                loc = np.intersect1d(np.where(xi1[:, 2] == 0), np.where(xi2[:, 2] == 3))
                xi1[loc, 2] = 4
                loc = np.intersect1d(np.where(xi1[:, 2] == 3), np.where(xi2[:, 2] == 0))
                xi2[loc, 2] = 4
                
            # Calculate the Euclidean distance
            r = np.sqrt((xi1 - xi2)**2)
            self._update_sg(r, firefly1.id, firefly2.id)
            # Calculate the attractiveness
            beta = self.beta0[0] * np.exp(-self.gamma(t,0) * r**2) 
            # Update the position of firefly1
            noise = np.random.randn(*xi2.shape)
            noise[:, 0:2] = self.alpha(t,0) * noise[:, 0:2]    # Maybe adjust parameter to be different for rotation
            
            if xi1.shape[1] == 3: # If rotatable
                beta[:,2] = self.beta0[1] * np.exp(-self.gamma(t,1) * r[:,2]**2) 
                noise[:, 2] = self.alpha(t,1) * noise[:, 2]

            xi = xi1 + beta * (xi2 - xi1) + noise   
            X.append(xi)
        valid = firefly1.set_X(X)
        if not valid:
            raise ValueError("Invalid position") # Should avoid this case

    def optimize(self, optimize_type = "taxi_cab_dist") -> Room:
        """
        Optimizes the room layout using the Firefly Algorithm (FA).
        :param optimize_type: Type of evaluation function to use. Options are "taxi_cab_dist" or "open_dist".
        :return: Best room layout found
        """
        # Evaluate the objective function for each firefly
        for firefly in self.fireflies:
            firefly.evaluate(optimize_type)
        # Sort fireflies by objective function value
        self.fireflies.sort(key=lambda x: x.fobj, reverse=True)
        # Initialize the best solution
        best = self.fireflies[0]
        # Main loop
        for t in range(self.T):
            # Update fireflies
            for i in range(self.N):
                for j in range(self.N):
                    if self.fireflies[i].fobj < self.fireflies[j].fobj:
                        # Move firefly1 towards firefly2
                        self.move_firefly(self.fireflies[i], self.fireflies[j], t)     
                        # Evaluate the objective function
                        self.fireflies[i].evaluate(optimize_type)
            # Sort fireflies by objective function value
            self.fireflies.sort(key=lambda x: x.fobj, reverse=True)
            # Update the best solution
            if self.fireflies[0].fobj > best.fobj:
                best = self.fireflies[0]
        return best.room



if __name__ == "__main__":
    # Example usage of the Firefly Algorithm (FA)
    table1 = Object(10, 10, 5, "Table") 
    couch1 = Object(30, 10, 8, "Couch")
    desks = [Object(20, 10, 5, "Desk") for i in range(5)]
    door1 = Object(10, 0, 8, "Door", x=20, y=0, rotation=Rotation.UP, rotatable=False, moveable=False)

    width = 100
    height = 100
    objects = [table1, couch1, door1] + desks#, temp1, temp2]
    N = 300  # Number of fireflies
    T = 20  # Number of iterations

    FA = FA(objects, width, height, N, T)
    room = FA.optimize("taxi_cab_dist") 

    print(room)
    print(f"Objective function value: {room.fobj}")
    print(f"Solution: {room.get_X()}")

    plt.subplot(1, 2, 1)
    plt.imshow(room.uid_map(), origin='lower')
    plt.title("UID Space")

    plt.subplot(1, 2, 2)
    plt.imshow(room.open_map(), origin='lower')
    plt.title("Open Space")

    plt.show()