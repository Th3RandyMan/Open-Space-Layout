from FA import FA
from Storage import Object, Rotation, Room
from matplotlib import pyplot as plt
import numpy as np

def plot(room: Room):
    plt.imshow(room.open_map(), origin='lower')
    plt.title(f"Open Space")
    plt.show()

if __name__ == "__main__":
    desks = [Object(5, 2, 3, "Desk") for i in range(10)]   # Create a list of desks
    shelves = [Object(3, 1, 4, "Shelf") for i in range(2)]
    Cabinets = [Object(3, 2, 2, "Cabinets") for i in range(3)]
    Couch = Object(6, 2, 3, "Couch")
    Tables = [Object(4, 2, 3, "Table") for i in range(3)]

    door1 = Object(4, 0, 4, "Door", x=0, y=0, rotation=Rotation.UP, rotatable=False, moveable=False)
    door2 = Object(width=4, depth=0, reserved_space=4, name="Door", x=14, y=0, rotation=Rotation.UP, rotatable=False, moveable=False)

    width = 40     # Width of the room
    height = 20    # Height of the room
    objects = [door1, door2, Couch] + desks + Tables
    #objects = desks
    N = 20  # Number of fireflies
    T = 30  # Number of iterations

    FireflyAlgorithm = FA(objects, width, height, N, T, name="LEPS Room")
    room = FireflyAlgorithm.optimize("taxi_cab_dist") 

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
    a = 5