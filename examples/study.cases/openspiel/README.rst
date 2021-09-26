Reinforcement Learning examples on Openspiel
==============================================

This folders contain a ready-to-use setup to run Openspiel games. 

Pre-Requisites:
------------------

To use MuJoCo environments, follow the installation instructions here: https://github.com/openai/mujoco-py#install-mujoco. For this, you will first need to acquire a `MuJoCo license <https://www.roboti.us/license.html>`_
To use PyBullet environments, simply run the :code:`./install_deps.sh` script. PyBullet requires no license to run.

Running an environment:
-------------------------

Any of the following environments are available for testing:

.. code-block:: bash
   
   % MuJoCo
   Ant-v2
   HalfCheetah-v2
   Hopper-v2
   Humanoid-v2
   HumanoidStandup-v2
   InvertedDoublePendulum-v2
   InvertedPendulum-v2
   Reacher-v2
   Swimmer-v2
   Walker2d-v2
   
   % PyBullet
   AntBulletEnv-v0
   HalfCheetahBulletEnv-v0
   HopperBulletEnv-v0
   HumanoidBulletEnv-v0
   Walker2DBulletEnv-v0

To run any of these, use the following example:

.. code-block:: bash

   python3 run-vracer.py --env AntBulletEnv-v0

Producing a movie:
-------------------------

To generate a movie that displays the outcome of a particular trained policy, use the following command:

.. code-block:: bash

   python3 ./genMovie --env AntBulletEnv-v0 --input _result_vracer_AntBulletEnv-v0 --output myMovie
   
The command will read the result of training an `AntBulletEnv-v0` environment from the `_result_vracer_AntBulletEnv` folder and output an `.mp4` movie in the `myMovie` folder.

