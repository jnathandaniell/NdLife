from numpy import ndarray, zeros, s_, sum
from numpy.random import default_rng
from numpy.lib.stride_tricks import sliding_window_view
from itertools import product


low = 1/4
mid = 3/8
high = 1/2


class Grid():
    """
    Grid class for generalized N-dimensional Conway's Game of Life.

    Follows the rules of Conway's Game of Life using approximate neighbor
    binning to support automatic runs in higher dimensional space.

    After generation, execute steps using the '.execute()' or
    '.wrapped_execute()' functions for cases when onde desires the grid to
    wrap around both sides.

    Execution handled by the 'sliding_window_view()' function from numpy's
    lib.stride_tricks module to generate many neighbor views, then accessed
    by index to generate a new array (should be more efficient than general
    element-wise operations).

    **kwargs:
        grid (numpy.ndarray): A user-provided initial grid space.
        wrap (bool): Default for whether the grid is 'wrapped' or not.
        rate (float): Default chance for a cell to be live for an automatically
            generated grid.
    """

    def __init__(self, dims: tuple, **kwargs) -> None:
        self.dims = dims
        self.wrap = False
        self.rate = 0.5
        grid_provided = False
        for key, arg in kwargs.items():
            match key:
                case 'grid':
                    self.grid = arg
                    self.dims = self.grid.shape
                    grid_provided = True
                case 'wrap':
                    self.wrap = arg
                case 'rate':
                    self.rate = arg
        if not grid_provided:
            self.make_grid()
        self.get_unit_dims()
    
    def make_grid(self) -> None:
        """
        Makes a basic random grid with user provided dimensions.
        """
        rng = default_rng()
        grid = rng.random(self.dims)
        grid[grid >= 1-self.rate] = True
        grid[grid < 1-self.rate] = False
        self.grid = grid
    
    def pad_grid(self) -> None:
        """
        Pads the grid.
        """
        padded_dims = tuple(i + 2 for i in self.dims)
        padded_grid = zeros(padded_dims)
        middle_slice = tuple(s_[1:-1] for _ in self.dims)
        padded_grid[middle_slice] = self.grid
        self.padded_grid = padded_grid
    
    def get_side_indices(self) -> None:
        """
        Gets the indices for the sides of a grid.

        Seems to be giving correct slices?.
        """
        sides = []
        pad_sides = []
        for i in self.dims:
            x = []
            y = []
            x_pad = []
            y_pad = []
            for j in self.dims:
                if j == i:
                    x.append(s_[1:-1])
                    x_pad.append(s_[1:-1])
                    y.append(s_[1:-1])
                    y_pad.append(s_[1:-1])
                else:
                    x.append(s_[1])
                    x_pad.append(s_[0])
                    y.append(s_[-2])
                    y_pad.append(s_[-1])
            x = tuple(s_[i] for i in x)
            y = tuple(s_[i] for i in y)
            x_pad = tuple(s_[i] for i in x_pad)
            y_pad = tuple(s_[i] for i in y_pad)
            sides.append((x, y))
            pad_sides.append((x_pad, y_pad))
        self.sides = tuple(sides)
        self.pad_sides = tuple(pad_sides)
    
    def wrap_grid(self) -> None:
        """
        Wraps the grid.

        Maybe working?.
        """
        self.get_side_indices()
        self.wrapped_grid = self.padded_grid
        for side, pad_side in zip(self.sides, self.pad_sides):
            self.wrapped_grid[pad_side[0]] = self.wrapped_grid[side[1]]
            self.wrapped_grid[pad_side[1]] = self.wrapped_grid[side[0]]
    
    def get_unit_dims(self) -> None:
        """
        Makes a unit grid template.
        """
        self.unit_dims = tuple(3 for _ in self.dims)
        self.unit_center = tuple(1 for _ in self.dims)
        self.unit_size = zeros(self.unit_dims).size - 1
    
    def get_window_indices(self) -> None:
        """
        Gets the tuple of unit window indices.
        """
        dim_vector = tuple([j for j in range(i)] for i in self.dims)
        self.views = tuple(product(*dim_vector))
    
    def get_windows(self) -> None:
        """
        Gets the windows of the correct dimension from the sliding window.
        """
        self.windows = sliding_window_view(self.padded_grid, self.unit_dims)
    
    def check_life(self, unit) -> bool:
        """
        Checks if a cell is live for next step.

        Check conditions for validity later.
        """
        center = unit[self.unit_center]
        neighbors = sum(unit) - center
        frac = neighbors / self.unit_size
        if frac < low: return False
        if frac > mid: return False
        if center and frac >= low and frac <= mid: return True
        if not center and frac >= mid and frac < high: return True
        return False
    
    def execute(self) -> ndarray:
        """
        Executes a step.
        """
        # Set up operation.
        self.pad_grid()
        self.get_windows()
        self.get_window_indices()
        new_grid = zeros(self.dims)
        # Execute operation.
        for index in self.views:
            new_grid[index] = self.check_life(self.windows[index])
        self.grid = new_grid
        return new_grid
    
    def execute_wrapped(self) -> ndarray:
        """
        Executes using wrapping.

        Maybe working?.
        """
        self.pad_grid()
        self.wrap_grid()
        self.padded_grid = self.wrapped_grid
        self.get_windows()
        self.get_window_indices()
        new_grid = zeros(self.dims)
        # Execute operation.
        for index in self.views:
            new_grid[index] = self.check_life(self.windows[index])
        self.grid = new_grid
        return new_grid

