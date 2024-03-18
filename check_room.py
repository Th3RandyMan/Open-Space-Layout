from Storage import Object, Room, Rotation
import numpy as np

room = Room(100, 100, "Room")

table1 = Object(10, 10, 5, "Table") 
couch1 = Object(30, 10, 8, "Couch")
desks = [Object(20, 10, 5, "Desk") for i in range(5)]
door1 = Object(10, 0, 8, "Door", x=20, y=0, rotation=Rotation.UP, rotatable=False, moveable=False)
objects = [table1, couch1, door1] + desks

loc = [np.array([[ 0., 77.,  1.]]), np.array([[ 0., 46.,  1.]]), np.array([[ 0., 21.,  0.],
       [34.,  1.,  0.],
       [56.,  5.,  2.],
       [74., 25.,  2.],
       [81., 65.,  1.]])]
loc = [l.astype(int) for l in loc]

i = 0
for l in loc:
    for x, y, r in l:
        while i < len(objects) and not objects[i].moveable:
            room.add_object(objects[i], objects[i].x, objects[i].y, objects[i].rotation)
            i += 1
        room.add_object(objects[i], x, y, r)
        i += 1

print(room)
print("Objective function value: ",room.evaluate("taxi_cab_dist"))
print(f"Solutions: {room.get_X()}")