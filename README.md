bsptran
=======

Apply affine transformations (scaling, rotation, reflection) to Quake 2 BSP
files without recompilation.

Requirements
------------

* Python ≥ 3.5
* NumPy ≥ 1.10

Usage
-----

`bsptran.py [options] <input.bsp> <output.bsp>`

Options
-------

Options can be specified multiple times. No options mean identity transformation.

* `-s <scale>` — scale the map on all axes
* `-s[xyz] <scale>` — scale the map on x/y/z axis (can be negative for reflection)
* `-r[xyz] <angle>` — rotate specified amount of degrees around x/y/z axis

Example
-------

Make the map twice as big:

    bsptran.py -s 2 i.bsp o.bsp

Reflect the map on y axis:

    bsptran.py -sy -1 i.bsp o.bsp

Reflect on x axis, scale by factor of 2 on y axis, then rotate 90 degrees
around x axis:

    bsptran.py -sx -1 -sy 2 -rx 90 i.bsp o.bsp

Caveats
-------

* It is very easy to generate a map that will crash software renderer.
* Entity string transformation code is incomplete.
* Expect func\_ entities to misbehave after rotation.
* Lightmaps can get occasionally corrupted depending on transformation (known
  Quake 2 bug).
