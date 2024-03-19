from FA import FA
from Storage import Object, Rotation, Room
from matplotlib import pyplot as plt
import numpy as np

def plot(room: Room):
    plt.imshow(room.open_map(), origin='lower')
    plt.title(f"Open Space")
    plt.show()

if __name__ == "__main__":
    table1 = Object(10, 10, 5, "Table") 
    couch1 = Object(30, 10, 8, "Couch")
    desk1 = Object(20, 10, 5, "Desk")
    desks = [Object(20, 10, 5, "Desk") for i in range(5)]   # Create a list of desks
    door1 = Object(10, 0, 8, "Door", x=20, y=0, rotation=Rotation.UP, rotatable=False, moveable=False)
    door2 = Object(width=10, depth=0, reserved_space=8, name="Door", x=70, y=0, rotation=Rotation.UP, rotatable=False, moveable=False)

    width = 100     # Width of the room
    height = 100    # Height of the room
    objects = [table1, couch1, door1, door2] + desks
    N = 10  # Number of fireflies
    T = 10  # Number of iterations

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