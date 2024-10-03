from dataclasses import dataclass
import numpy as np
import curses
from time import sleep


@dataclass
class LifeSpace():
    """Space for the 2D Conway's Game of Life."""

    rng = np.random.default_rng()
    dim_1 = rng.integers(1, 20, size=1, endpoint=True)[0]
    dim_2 = rng.integers(1, 20, size=1, endpoint=True)[0]
    space = rng.random(size=(dim_1, dim_2))
    space[space >= 0.8] = True
    space[space < 0.8] = False

    def __init__(self, *,
                 space=space,
                 wrap=False,
                 track=False):
        self.space = space
        self.dims = np.shape(self.space)
        self.wrap = wrap
        self.track = track
        # Make an index space.
        self.ind_list = [(i, j)
                         for i in range(self.dims[0])
                         for j in range(self.dims[1])]
        self.hist = [self.space]

    def check_valid_index(self, index):
        """Checks if an index location is valid."""
        # If any dimensional index is less than 0, return False.
        if any(index) < 0:
            return False
        # If any dimensional index exceeds the size of the dimension,
        # return False
        for n, d in zip(index, self.dims):
            if n > d-1:
                return False
        # Return True (valid index).
        return True
    
    def wrap_switch(self, n, d):
        """Simple switch for wrapper."""
        # Wrap on low end to high end.
        if n < 0:
            return d-1
        # Wrap on high end to low end.
        elif n > d-1:
            return 0
        # Return the current location otherwise.
        return n
    
    def wrap_index(self, index):
        """Gets the wrapped index."""
        return tuple(self.wrap_switch(n, d)
                     for n, d, in zip(index, self.dims))

    def get_neighbors(self, index):
        """Gets the neighbor indices of a cell."""
        # Get ranges of each dimension.
        ranges = [(n-1, n, n+1) for n in index]
        # Get permutations for neighbor locations.
        neighbor_ind = [(a, b) for a in ranges[0] for b in ranges[1]]
        # Remove own cell.
        neighbor_ind.remove(index)
        # Operate with wrapping.
        if self.wrap:
            return tuple(self.wrap_index(i) for i in neighbor_ind)
        # Collect the valid indices for neighbors (nothing out of bounds).
        return tuple(i for i in neighbor_ind if self.check_valid_index(i))
    
    def count_live_neighbors(self, neighbors):
        """Gets the nnumber of live neighbors for a cell based on neighbors."""
        return sum([self.space[neighbor] for neighbor in neighbors])

    def check_live(self, live, n_count):
        """Checks if a cell is alive after this step."""
        # If the cell is alive, kill it if it does not have 2 or 3 neighbors.
        if live:
            if n_count == 2 or n_count == 3:
                return True
            else:
                return False
        # If the cell is dead, give it life if it has 3 neighbors.
        else:
            if n_count == 3:
                return True
        return False
    
    def get_next_cell(self, index):
        """Gets the next value for a single cell."""
        # Get the neighbors.
        neighbors = self.get_neighbors(index)
        # Count the live neighbors.
        live_neighbors = self.count_live_neighbors(neighbors)
        # Return the bool value of the current cell.
        return self.check_live(self.space[index], live_neighbors)
    
    def get_next_space(self):
        """Gets the space for the next step on the board."""
        # Perform the operation based on indices.
        new_space = np.empty(shape=self.dims)
        for index in self.ind_list:
            new_space[index] = self.get_next_cell(index)
        return new_space

    def step(self):
        """Takes a step in the game."""
        self.space = self.get_next_space()
        if self.track:
            self.hist.append(self.space)
        return self.space


class LifeViewer():
    """Viewer unit for 2D game of life."""

    life_space = LifeSpace()
    g = 100
    framerate = 10

    def __init__(self, *, life_space=life_space, g=g, framerate=framerate):
        self.life_space = life_space
        self.g = g
        self.framerate = framerate

    def get_view(self, grid):
        """Gets the string representation of a grid."""
        repr = ''
        # For each row in the grid.
        for i in grid:
            # For each column in the grid.
            for j in i:
                # Give a symbol if cell is alive.
                if j:
                    repr += '0'
                # Give white space if cell is dead.
                if not j:
                    repr += ' '
            # Make a newline when the end of a line is reached.
            repr += '\n'
        return repr

    def _draw(self, screen, g):
        """Draws a grid."""
        # Prepare the terminal for the representation.
        curses.curs_set(0)
        screen.clear()
        # Set the initial view.
        try:
            screen.addstr(0,0, self.get_view(self.life_space.space))
        except curses.error:
            raise ValueError('Error: terminal too small')
        # Iterate over the requested number of generations.
        for _ in range(g):
            try:
                grid = self.life_space.step()
                screen.addstr(0,0, self.get_view(grid))
                screen.refresh()
                sleep(1 / self.framerate)
            # Allow keyboard interrupts.
            except KeyboardInterrupt:
                print("Terminated.")
                break

    def show(self):
        """Show the life space."""
        # Initialize the screen.
        stdscr = curses.initscr()
        # Run the method.
        try:
            curses.wrapper(self._draw(stdscr, self.g))
        # Close the window when interrupted/ over.
        except:  # noqa: E722
            curses.endwin()
        else:
            curses.endwin()

        