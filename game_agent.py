"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def __valid_jumps(loc, game):
    """Generate the list of possible moves for an L-shaped motion (like a
    knight in chess).
    """
    if loc == game.NOT_MOVED:
        return game.get_blank_spaces()

    r, c = loc
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                    if game.move_is_legal((r + dr, c + dc))]
    random.shuffle(valid_moves)
    return valid_moves

def custom_seek_sum_movements(game, player):
    # When Improved score (from lecture) considers legal moves left, this heuristic
    # take into account, the sum of total number of leaps the knight can do within
    # all of its remaining legal moves. In case of tied legal moves, this helps identify
    # the player with most reachable open spaces left on the board.
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    len_own = float(len(own_moves))
    len_opp = float(len(opp_moves))

    own_movements = float(sum([len(__valid_jumps(move, game)) for move in own_moves ]))
    opp_movements = float(sum([len(__valid_jumps(move, game)) for move in opp_moves ]))
    return own_movements - opp_movements

    # if len_own == len_opp:
    #     own_movements = float(sum([len(__valid_jumps(move, game)) for move in own_moves ]))
    #     opp_movements = float(sum([len(__valid_jumps(move, game)) for move in opp_moves ]))
    #     return own_movements - opp_movements
    
    # return len_own - len_opp


def custom_seek_average_movements(game, player):
    # Builds on custom_seek_sum_movements. When the movements are tied, 
    # who has the more number of open movements per move, helps.
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    len_own = float(len(own_moves))
    len_opp = float(len(opp_moves))

    own_movements = float(sum([len(__valid_jumps(move, game)) for move in own_moves ]))
    opp_movements = float(sum([len(__valid_jumps(move, game)) for move in opp_moves ]))

    
    if len_own == 0:
        return float("-inf")
    elif len_opp == 0:
        return float("inf")
    else:
        return own_movements/len_own - opp_movements/len_opp

def custom_seek_center_position(game,player):
    # Consider positional advantage by determining who is nearer to the center of the board. Being closer to center 
    # has more freedom of movement to expand out than being at edge/corner.

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_y, center_x = int(game.height / 2), int(game.width / 2)
    own_y, own_x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(game.get_opponent(player))
    own_distance = abs(own_y - center_y) + abs(own_x - center_x)
    opp_distance = abs(opp_y - center_y) + abs(opp_x - center_x)
    return float(opp_distance - own_distance) / 10.

def custom_seek_movements_positions(game,player):
    # Apart from having the sum_movements, consider positional advantage as well
    # by determining who is nearer to the center of the board. Being closer to center 
    # has more freedom of movement to expand out than being at edge/corner.

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_movements = float(sum([len(__valid_jumps(move, game)) for move in own_moves ]))
    opp_movements = float(sum([len(__valid_jumps(move, game)) for move in opp_moves ]))
    
    center_y, center_x = int(game.height / 2), int(game.width / 2)
    own_y, own_x = game.get_player_location(player)
    opp_y, opp_x = game.get_player_location(game.get_opponent(player))
    own_distance = abs(own_y - center_y) + abs(own_x - center_x)
    opp_distance = abs(opp_y - center_y) + abs(opp_x - center_x)
    return float(own_movements - opp_movements) / 10. + float(opp_distance - own_distance) / 10.

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return custom_seek_sum_movements(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.
        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # If no legal mmoves, resign
        if not legal_moves:
            return (-1, -1)

        # Take center if not taken.
        # center = (int(game.height/2), int(game.width/2))
        # if center in legal_moves:
        #     return center

        if game.move_count == 0:
            return(int(game.height/2), int(game.width/2))

        # default move if we run out of time
        best_move = legal_moves[0]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative == True:
                idepth = 1
                if self.method == 'minimax':
                    while True:
                        best_score, best_move = self.minimax(game, idepth)
                        #gameover check
                        if best_score == float("inf") or best_score == float("-inf"):
                            break
                        idepth += 1
                elif self.method == 'alphabeta':
                    while True:
                        best_score, best_move = self.alphabeta(game, idepth)
                        if best_score == float("inf") or best_score == float("-inf"):
                            break
                        idepth += 1
                else:
                    raise ValueError('Invalid method')
            else:
                if self.method == 'minimax':
                    x, best_move = self.minimax(game, self.search_depth)
                elif self.method == 'alphabeta':
                    x, best_move = self.alphabeta(game, self.search_depth)
                else:
                    raise ValueError('Invalid method')

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move


    

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        default_lowest = float("inf")
        default_highest = float("-inf")
        default_move = (-1,-1)
        best_move = default_move
        lowest_score, highest_score = default_lowest, default_highest

        # If no more legal moves left for the player, return resignation.
        moves_for_ply = game.get_legal_moves()
        if len(moves_for_ply) == 0:
            if maximizing_player == True:
                return default_highest, default_move
            else:
                return default_lowest, default_move

        # If desired depth or terminal is reached.
        if depth == 1:
            if maximizing_player == True:
                for move in moves_for_ply:
                    score = self.score(game.forecast_move(move), self)
                    # win for MAX player, quit searching
                    if score == float("inf"):
                        return score, move
                    #else, keep searching for max score
                    elif score > highest_score:
                        highest_score, best_move = score, move
                return highest_score, best_move
            else:
                for move in moves_for_ply:
                    score = self.score(game.forecast_move(move), self)
                    # win for MIN player, quit searching
                    if score == float("-inf"):
                        return score, move
                    #else, keep searching for min score
                    elif score < lowest_score:
                        lowest_score, best_move = score, move
                return lowest_score, best_move

        if maximizing_player == True:
            for move in moves_for_ply:
                score, x = self.minimax(game.forecast_move(move), depth-1, maximizing_player = False)
                # win for MAX player, quit searching
                if score == float("inf"):
                    return score, move
                #else, keep searching for max score    
                elif score > highest_score:
                    highest_score, best_move = score, move
            return highest_score, best_move
        else:
            for move in moves_for_ply:
                score, x = self.minimax(game.forecast_move(move), depth-1, maximizing_player=True)
                # win for MIN player, quit searching
                if score == float("-inf"):
                    return score, move
                #else, keep searching for min score
                if score < lowest_score:
                    lowest_score, best_move = score, move
            return lowest_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)
        Returns
        -------
        float
            The score for the current search branch
        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        
        default_alpha = float("inf")
        default_beta = float("-inf")
        default_move = (-1,-1)
        best_move = default_move
        lowest_alpha, highest_beta = default_alpha, default_beta

        # If no more legal moves left for the player, return resignation.
        moves_for_ply = game.get_legal_moves()
        if len(moves_for_ply) == 0:
            if maximizing_player == True:
                return default_beta, default_move
            else:
                return default_alpha, default_move

        # if desired depth or leaf node is reached, evaluate the score
        if depth == 1:
            if maximizing_player == True:
                for move in moves_for_ply:
                    score = self.score(game.forecast_move(move), self)
                    #score >= beta, tree can be pruned - donot enter remaining nodes/moves
                    if score >= beta:
                        return score, move
                    #else keep going, keeping track of best move
                    elif score > highest_beta:  
                        highest_beta, best_move = score, move
                return highest_beta, best_move
            else:
                for move in moves_for_ply:
                    score = self.score(game.forecast_move(move), self)
                    #score <= alpha, tree can be pruned - donot enter remaining nodes/moves
                    if score <= alpha:
                        return score, move
                    #else keep going, keeping track of best score & move
                    elif score < lowest_alpha:
                        lowest_alpha, best_move = score, move
                return lowest_alpha, best_move

        # while in recursion (depth >1)
        if maximizing_player == True:
            for move in moves_for_ply:
                score, x = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, maximizing_player = False)
                #score >= beta, tree can be pruned - donot enter remaining nodes/moves
                if score >= beta:
                    return score, move
                #else keep going, keeping track of best score & move
                elif score > highest_beta:
                    highest_beta, best_move = score, move
                alpha = max(alpha, highest_beta)
            return highest_beta, best_move
        else:
            for move in moves_for_ply:
                score, x = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, maximizing_player=True)
                #score <= alpha, tree can be pruned - donot enter remaining nodes/moves
                if score <= alpha:
                    return score, move
                #else keep going, keeping track of best score & move
                elif score < lowest_alpha:
                    lowest_alpha, best_move = score, move
                beta = min(beta, lowest_alpha)
            return lowest_alpha, best_move