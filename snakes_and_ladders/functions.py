import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# This was taken from the example in https://scipython.com/book/chapter-6-numpy/additional-problems/analysing-snakes-and-ladders-as-a-markov-chain/

def create_transition_matrix(snakes, ladders):
    transition = ladders + snakes

    # Set up the transition matrix
    T = np.zeros((101, 101))
    for i in range(1, 101):
        T[i - 1, i:i + 6] = 1 / 6

    for (i1, i2) in transition:
        iw = np.where(T[:, i1] > 0)
        T[:, i1] = 0
        T[iw, i2] += 1 / 6

    # House rules: you don't need to land on 100, just reach it.
    T[95:100, 100] += np.linspace(1 / 6, 5 / 6, 5)
    for snake in snakes:
        T[snake, 100] = 0

    return T

def current_position(position):
    v = np.zeros(101)
    v[position] = 1
    print(v)
    return v

def possible_next_position(v,T):
    v = v.dot(T)
    return np.nonzero(v)


def determine_modal_moves(v,T):
    n, P = 0, []
    cumulative_prob = 0
    # Update the state vector v until the cumulative probability of winning
    # is "effectively" 1
    while cumulative_prob < 0.99999:
        n += 1
        v = v.dot(T)
        P.append(v[100])
        cumulative_prob += P[-1]
    mode = np.argmax(P) + 1
    print('modal number of moves:', mode)

    # Plot the probability of winning as a function of the number of moves
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, n, n), P, 'g-', lw=2, alpha=0.6, label='Markov')
    ax.set_xlabel('Number of moves')
    ax.set_ylabel('Probability of winning')



class Square:
    def __init__(self):
        self.position = []
        self.base_colour = 0
        self.colour = self.base_colour
        self.is_snake = False
        self.is_ladder = False
        self.occupied_by = 'nobody'
        self.occupied_colors= {'nobody': self.base_colour, 'player1':4, 'player2':5}

    @property
    def base_colour(self):
       return self.base_colour

    @base_colour.setter
    def base_colour(self, value: int):
        self.base_colour = value

    @property
    def colour(self):
        return self.base_colour

    @base_colour.setter
    def colour(self, value: int):
        self.colour = value

    @property
    def position(self):
        return self.position

    @position.setter
    def position(self, value: Tuple):
        self.base_colour = value

    @property
    def is_snake(self):
        return self.is_snake

    @is_snake.setter
    def is_snake(self, value: bool):
        self.is_snake = value
        self.base_colour = 2

    @property
    def is_ladder(self):
        return self.is_ladder

    @is_snake.setter
    def is_ladder(self, value: bool):
        self.is_ladder = value
        self.base_colour = 3

    @property
    def occupied_by(self):
        return self.occupied_by

    @occupied_by.setter
    def occupied_by(self, value: str):
        self.occupied_by = value
        if value != 'nobody':
            self.colour = self.occupation_colors[str]


class Board:
    def __init__(self, x_dim=10, y_dim=10):
        self.board = []
        self.x_dim = x_dim
        self.y_dim = y_dim

    @property
    def board(self):
        return self.board

    @board.setter
    def board(self, x_dim, y_dim, snakes=[], ladders=[]):
       return  self.create_board(x_dim, y_dim, snakes, ladders)


    def create_board(self, x_dim=10, y_dim=10, snakes=[], ladders=[]):
        X = np.linspace(1,x_dim)
        Y = np.linspace(1,y_dim)

        board = []
        for x in X:
            for y in Y:
                square = Square.position(x,y)
                if (x % 2) == 0 & (y % 2) ==0 | (x % 2) != 0 & (y % 2) !=0 :
                    square.base_colour(0)
                else:
                    square.base_colour(1)
                board.append(square)

        if snakes:
            for sq in board:
              for snake in snakes:
                    if sq.poition == snake:
                        sq.is_snake(True)

        if ladders:
            for sq in board:
              for ladder in ladders:
                if sq.poition == ladder:
                    sq.is_ladder(True)


        return board


    def board(self):
        display_board = np.zeros(self.x_dim, self.y_dim)
        for sq in self.board:
            display_board[sq.position[0], sq.position[1]] = sq.colour

        plt.imshow(display_board)
