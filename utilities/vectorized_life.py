#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vectorized version of Conway's Game of Life.

Props to http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/ for
coming up with some of the concepts in vectorization of boards."""


from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import as_strided


@dataclass
class VectorLife:
    """Vectorized version of Conway's Game of Life."""

    rng = np.random.default_rng()
    dims = (20, 20)
    board = rng.random(size=dims)
    board[board >= 0.8] = True
    board[board < 0.8] = False
    
    def __init__(self, *, board, wrap=False, track=False) -> None:
        self.board = board
        self.dims = np.shape(board)
        self.wrap = wrap
        self.track = track
        if self.track:
            self.tape = [board]
        else:
            self.tape = None
        pass
    
    def step(self):
        """Make a single step."""
        pass

    def run(self, g: int):
        """Runs the Game of Life."""
        for _ in range(g):
            try:
                self.step()
                if self.track:
                    self.tape.append(self.board)
            except KeyboardInterrupt:
                print('Terminating.')
                break

