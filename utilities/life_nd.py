#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Multi-dimensional Conway's Game of Life.

As Fred would have me call it, Jimmy's nD Game of Life.
Treats the number of neighbors as a step function to make generalized rules.
Current focus is to search for still lifes and oscillators in any number of
dimensions.
Leverages element-wise operations due to difficulty with vectorized
indexing.
Currently need to develop the index list, difficulties expanding to nD."""


from math import pow
import numpy as np


class LifeND():
    """Class for the n-dimensional Conway's Game of Life."""

    ############################################
    ##### Default param generator and init #####
    ############################################
    dims = 2
    shape = tuple(3 for _ in range(dims))
    space = np.ones(shape=shape)

    def __init__(self, *, space=space, wrap=False, track=True) -> None:
        # Generate initial space, dimensions, and rules.
        self.space = space
        self.dims = np.shape(space)
        self.make_rules()
        self.wrap = wrap
        # Make an index space (NOT CORRECT CURRENTLY).
        self.ind_list = [tuple(a, b, c, ... n)
                         for a in range(self.dims[0])
                         for b in range(self.dims[1])
                         for c in range(self.dims[2])
                         ...
                         for n in range(self.dims[-1])]
        # History tracking (as runtape).
        self.track = track
        if self.track:
            self.tape = [self.space]

    #######################################
    ##### Index validity and wrapping #####
    #######################################
    def check_valid_index(self, index) -> bool:
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
    
    def wrap_switch(self, n, d) -> int:
        """Simple switch for wrapper."""
        # Wrap on low end to high end.
        if n < 0:
            return d-1
        # Wrap on high end to low end.
        elif n > d-1:
            return 0
        # Return the current location otherwise.
        return n
    
    def wrap_index(self, index) -> tuple:
        """Gets the wrapped index."""
        return tuple(self.wrap_switch(n, d)
                     for n, d, in zip(index, self.dims))
    
    ###############################
    ##### Neighbor Collection #####
    ###############################
    def get_neighbors(self, index) -> tuple:
        """Gets the neighbor indices of a cell."""
        # Get ranges of each dimension.
        ranges = [(n-1, n, n+1) for n in index]
        # Get permutations for neighbor locations.
        neighbor_ind = [tuple(a, b, c, d, ..., n)
                        for a in ranges[0]
                        for b in ranges[1]
                        for c in ranges[2]
                        for d in ranges[3]
                        ...
                        for n in ranges[len(self.dims-1)]]
        # Remove own cell.
        neighbor_ind.remove(index)
        # Operate with wrapping (NOT WORKING CURRENTLY!).
        if self.wrap:
            return tuple(self.wrap_index(i) for i in neighbor_ind)
        # Collect the valid indices for neighbors (nothing out of bounds).
        return tuple(i for i in neighbor_ind if self.check_valid_index(i))
    
    def count_live_neighbors(self, neighbors) -> int:
        """Gets the nnumber of live neighbors for a cell based on neighbors."""
        return sum([self.space[neighbor] for neighbor in neighbors])
    
    ###############################
    ##### Rule implementation #####
    ###############################
    def make_rules(self) -> None:
        """Makes the ruleset for the game, treating neighbors as step func."""
        # Calculate the proportionality rules for Conway's Game of Life.
        neighbors_l = 1/4
        neighbors_h = 3/8
        res_h = 1/2
        # Calculate the number of possible neighbors.
        num_neighbors = pow(3, self.dims) - 1
        # Generate the values of each neighbor in dims.
        ps = tuple(i / num_neighbors for i in range(num_neighbors))
        # Set flags for boundaries.
        low_found = False
        high_found = False
        res_high_found = False
        # Check the value of each neighbor in dims.
        for i in range(self.dims):
            # Check for the living lower bound.
            if not low_found:
                if ps[i] >= neighbors_l:
                    self.low_rule = i
                    low_found = True
            # Check for the living upper bound.
            if not high_found:
                if ps[i] == neighbors_h:
                    self.high_rule = i
                    self.res_low = i
                    high_found = True
                elif ps[i] > neighbors_h:
                    self.high_rule = i - 1
                    self.res_low = i - 1
                    high_found = True
            # Check for the resurrection upper bound.
            if not res_high_found:
                if ps[i] >= res_h:
                    self.res_high = i - 1
                    res_high_found = True
            # Break if all rules are found.
            if low_found and high_found and res_high_found:
                break

    def check_live(self, live, n_count):
        """Checks if a cell is alive after this step."""
        # If the cell is alive, kill it if it is not within rule range.
        if live:
            if n_count >= self.low_rule and n_count <= self.high_rule:
                return True
            else:
                return False
        # If the cell is dead, give it life if it is in resurrection range.
        else:
            if n_count >= self.res_low and n_count <= self.res_high:
                return True
        return False
    
    ###########################
    ##### Check functions #####
    ###########################
    def check_for_similarity(self, new_space) -> bool:
        """Checks to see if the space has already existed."""
        if any(self.tape) == new_space:
            return True
        return False
    
    def check_for_empty(self):
        """Checks to see if the space is empty."""
        if not any(self.space):
            return True
        return False

    ##################################
    ##### Cell and space updates #####
    ##################################
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
        new_space = self.get_next_space()
        # Check if the space is empty.
        if self.check_for_empty():
            return True, (self.space, None)
        # Check if the space has already existed.
        if self.check_for_similarity(new_space):
            a = self.tape.index(new_space)
            # Return the repeating shape as well as its period.
            return True, (new_space, len(self.tape) - a)
        # Update space.
        self.space = new_space
        if self.track:
            self.tape.append(self.space)
        return False, (self.space, None)
    
    def stable_search(self, t_max):
        """Takes multiple steps to search for still lifes and oscillators."""
        # Iterate over number of timesteps.
        i = 0
        while i < t_max:
            try:
                found, results = self.step()
                # Return the oscillator or still life if found.
                if found:
                    return results
                i += 1
            except KeyboardInterrupt:
                print(f'Terminated on generation {i}.')
                break

