Mountain Cart (Python)
======================

In this example the goal of the RL agent is to move a cart in a U-shaped valley as high as possible.
For that, it shall switch between applying left and right directed forces in order to benefit from gravity.


in _model/env.py set output = True and visualize states with script plotStates.py
use following command for a simplistic visualization of states


.. code-block:: bash

    ffmpeg -framerate 10 -pattern_type glob -i 'figures/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p cart.mp4
