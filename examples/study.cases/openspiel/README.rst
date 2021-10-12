Reinforcement Learning examples on Openspiel
==============================================

This folders contain a ready-to-use setup to run Openspiel games. 
Turn-based multi-agent games, are implemented with a special "waiting" action.
Illegal moves are implemented with a negative reward.

Running an environment:
-------------------------

Any of the following environments are available for testing:

.. code-block:: bash
   
   backgammon
   battleship
   blackjack
   blotto
   breakthrough
   bridge
   bridge_uncontested_bidding
   catch
   chess
   cliff_walking
   clobber
   coin_game
   connect_four
   coop_box_pushing
   coop_to_1p
   coordinated_mp
   cursor_go
   dark_chess
   dark_hex
   dark_hex_ir
   deep_sea
   efg_game
   first_sealed_auction
   gin_rummy
   go
   goofspiel
   hanabi
   havannah
   hearts
   hex
   kriegspiel
   kuhn_poker
   laser_tag
   leduc_poker
   lewis_signaling
   liars_dice
   liars_dice_ir
   markov_soccer
   matching_pennies_3p
   matrix_cd
   matrix_coordination
   matrix_mp
   matrix_pd
   matrix_rps
   matrix_rpsw
   matrix_sh
   matrix_shapleys_game
   mfg_crowd_modelling
   mfg_crowd_modelling_2d
   misere
   negotiation
   nfg_game
   normal_form_extensive_game
   oh_hell
   oshi_zumo
   othello
   oware
   pentago
   phantom_ttt
   phantom_ttt_ir
   pig
   quoridor
   rbc
   repeated_game
   sheriff
   skat
   solitaire
   start_at
   stones_and_gems
   tarok
   tic_tac_toe
   tiny_bridge_2p
   tiny_bridge_4p
   tiny_hanabi
   trade_comm
   turn_based_simultaneous_game
   universal_poker

To run any of these, use the following example:

.. code-block:: bash

   python3 run-vracer.py --env tic_tac_toe
