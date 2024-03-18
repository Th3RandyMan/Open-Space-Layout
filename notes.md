# Recorded Notes for Project
Not all items on the to-do list will be addressed, but more objects need to be added to the LEPS Room.

## To-do list:
* Synthesize more data for testing.
* Fix OptimizeLayout.ipynb or delete it.
* Investigate different objective functions for algorithm.
    * As of right now, walls are considered as objects as well. Maybe ignore the walls and only create contour maps from objects.
    * Euclidean wasn't working properly before.
    * Open distance is untested.
* Adjust noise separation method by adding parameters for adjust noise.
    * Maybe add weighting to object noise by using travel distance.
* Look into adjust rotation changes in firefly movement.
    * Improved method would be increase likelihood of rotating the close the object gets to another solution.
    * Less likely to rotate if far away
* Fix timeout method for random room generation.
* Implement non-contiguous fixes in set_X method
    * Could look for shortest distance between clusters and try to move objects there.
        * Could introduce more issues with surrounding objects.
        * Need to promote enough space between clusters to walk through.
* Investigate different parameters for the Firefly Algorithm
* DbFA is almost completed, but average distance between each firefly needs to be calculated.
    * Issue is that objects have different distances. May be difficult to resolve this.

## Types of Conflicts
* All of these are resolved using noise separation method. If rotated or moved, noise separation will be performed.
1. Two objects didn't move, but rotation changed. Now overlap.
    1. Give priority to the object that didn't rotate. Noise search for the other object.
    2. If both rotated, noisy separation. Two noise searches at the same time.
2. One object didn't move, but the other did. Now overlap.
    1. Create noise search, increase noise if failed.
3. Two objects moved, and now overlap.

## LEPS Room 
* Room size: 40ft by 19.2ft
* Desk 1 size: 58in by 23in
* NEED TO ADD MORE ITEMS

## Results on test example:
Default hyperparameters are alpha = (2,0.2), bet0 = (5,0.5), and gamma = 0.01. Runtime follow format of hours:minutes:seconds
* For N = 10 and T = 10, 00:10 runtime:
``` 
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00102703
Solution: [array([[36.,  0.,  1.]]), array([[ 0., 64.,  1.]]), array([[ 5.,  0.,  3.],
       [85.,  0.,  1.],
       [47., 74.,  2.],
       [67., 68.,  1.],
       [85., 61.,  1.]])]
```

* For N = 10 and T = 20, 00:20 runtime:
```
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00104827
Solution: [array([[33.,  1.,  0.]]), array([[69., 82.,  0.]]), array([[ 2.,  0.,  1.],
       [49.,  6.,  0.],
       [80., 40.,  2.],
       [81., 56.,  1.],
       [45., 90.,  2.]])]
```

* For N = 10 and T = 40, 00:42 runtime:
```
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00113211
Solution: [array([[17., 90.,  2.]]), array([[82., 70.,  1.]]), array([[ 0.,  1.,  1.],
       [70.,  0.,  1.],
       [ 2., 73.,  1.],
       [78., 34.,  1.],
       [85., 13.,  1.]])]
```

* For N = 100 and T = 10, 21:09 runtime:
```
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00120348
Solution: [array([[ 8., 90.,  1.]]), array([[ 3., 52.,  1.]]), array([[ 0.,  0.,  1.],
       [29., 79.,  1.],
       [85.,  3.,  1.],
       [54., 90.,  2.],
       [82., 80.,  1.]])]
```

* For N = 100 and T = 20, 45:02 runtime:
```
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00119344
Solution: [array([[ 3., 89.,  1.]]), array([[70., 16.,  2.]]), array([[ 2.,  1.,  1.],
       [50.,  5.,  2.],
       [ 0., 78.,  2.],
       [23., 80.,  1.],
       [80., 90.,  2.]])]
```

* For N = 100 and T = 40, 1:29:11 runtime:
```
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00123605
Solution: [array([[90., 48.,  2.]]), array([[ 3., 16.,  1.]]), array([[ 0.,  5.,  2.],
       [ 0., 72.,  2.],
       [85.,  0.,  1.],
       [ 0., 90.,  2.],
       [90., 74.,  3.]])]
```

* For N = 300 and T = 20, 20:27:41 runtime:
```
Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00133779
Solution: [array([[61., 90.,  1.]]), array([[70.,  0.,  0.]]), array([[ 1.,  0.,  1.],
       [ 0., 21.,  1.],
       [ 2., 90.,  2.],
       [26., 90.,  2.],
       [79., 90.,  2.]])]
```
