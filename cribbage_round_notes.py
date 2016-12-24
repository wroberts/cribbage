# set up the current round
# set the go_flag to False
go_flag = False
# set the 31_flag to False
flag_31 = False

# loop forever:
while True:

    # exit the loop if the game is over (no player has cards left)
    if no_cards_left():
        break

    # given that it's player 1's turn

    # compute the legal moves available to player 1
    legal_moves = get_legal_moves()

    # if player 1 is in "Go" and has no legals moves, or player 1 has hit 31
    if (go_flag and not legal_moves) or (flag_31):
        # then player 1 gets awarded 1 or 2 points, and we restart the round
        if flag_31:
            award(player_1, 2)
        else:
            award(player_1, 1)
        restart_round()
        continue
            
    # if player 1 has no moves to make
    if not legal_moves:

        # player 1 says "Go"
        go()
        # set the go_flag to True
        go_flag = Truee
        # switch turns
        switch_players()
        # loop
        continue
        
    # otherwise
    else:

        # ask player 1 to choose a move
        move = choose_move()

        # make the move
        make_move(move)

        # if the move makes the count hit 31, set the 31 flag
        if score_count() == 31:
            flag_31 = True

        if not go_flag and not flag_31:
            # switch to other player
            switch_players()

        # loop
        continue
