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
Default hyperparameters are alpha = (2,0.2), bet0 = (5,0.5), and gamma = 0.01.
* For N = 10 and T = 10 on test example:
``` Room | Width: 100, Height: 100, Number of Objects: 8 Contiguous: True
Objects: ['Door: 1', 'Table: 1', 'Couch: 1', 'Desk: 5']
Objective function value: 0.00104568
Solution: [array([[5., 7., 3.]]), array([[ 0., 26.,  1.]]), array([[ 0., 57.,  1.],
       [66.,  2.,  1.],
       [ 2., 85.,  0.],
       [85., 35.,  1.],
       [80., 64.,  0.]])] 
```

* For N = 10 and T = 20 on test example:
```

```

* For N = 10 and T = 40 on test example:
```

```

* For N = 100 and T = 10 on test example:
```

```

* For N = 100 and T = 20 on test example:
```

```

* For N = 100 and T = 40 on test example:
```

```