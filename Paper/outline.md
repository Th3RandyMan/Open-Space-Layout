# Paper Outline
Feel free to ask me for help on any of this. You are welcome to change this up. 

Don't worry about numbering figures, I'll add in referencing in latex. After the draft is done, we will need to put all the figures in the figures folder. 

* Abstract 
    * Probably leave this for last. Either I can do it, you can, or we have other methods ;)
* Introduction
* Methodology (each could be subsection)
    * Could be called something else.
    * Firefly algorithm (and genetic if used)
    * Sum of the Contour Map for objective function
    * Contiguous constraint
    * Types of Conflicts and methods to resolve
* Design Choices
    * Objects being rectangles.
    * Random Room generation
    * Contour map rather than others.
        * Contour map could be done by Conor's method, euclidean, Taxicab (current implementation).
        * Each contour map could look at distance to each object or each object and the walls (current implementation).
    * Solution Vector
        * Dimensionality is the number of unique shapes (including reserved space)
        * Each element is a shape list (numpy array)
            * Each shape is ordered by closest to origin
        * Adjust rotations between 0 and 3 to be 4 and 3
    * How conflicts are resolved.
* Experimental Results - should test on 2 or 3 different rooms
    * Show uid plots before and after running the algorithm.
    * Specify different params
    * Watch one firefly as the room changes through iterations.
    * Show results of rooms for best solutions
* Conclusion
    * Talk about the results of the tests.
    * Discuss challenges
* Future works
    * Adjust to be convex shapes with full rotations.
        * Maybe round up on shape to get closer to rectangular edges.
    * Look into additional constraints that may help performance.
    * Add on visual for rooms to get better evaluation from designers.
* References
    * Firefly algorithm
    * 

