"""
Boids Flocking Model
===================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))


import numpy as np

from mesa import Model
from mesa.examples.basic.boid_flockers.agents import Boid
from mesa.experimental.continuous_space import ContinuousSpace


class BoidFlockers(Model):
    def __init__(
        self,
        population_size=100,  # total number of boids
        width=100,
        height=100,
        speed=1,
        vision=10,
        separation=2,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
            n_agents=population_size,
        )

        # Determine the number of blue and black boids
        blue_population = int(population_size * 0.10)   # 10% blue
        green_population = int(population_size * 0.25)   # 10% green
        black_population = population_size - blue_population - green_population 

        # Create blue boids (the ones that wander randomly)
        blue_positions = self.rng.random(size=(blue_population, 2)) * self.space.size
        blue_directions = self.rng.uniform(-1, 1, size=(blue_population, 2))
        Boid.create_agents(
            self,
            blue_population,
            self.space,
            position=blue_positions,
            direction=blue_directions,
            cohere=cohere,
            separate=separate,
            match=match,
            speed=speed,
            vision=vision,
            separation=separation,
        )

        green_positions = self.rng.random(size=(green_population, 2)) * self.space.size
        green_directions = self.rng.uniform(-1, 1, size=(green_population, 2))
        Boid.create_agents(
            self,
            green_population,
            self.space,
            position=green_positions,
            direction=green_directions,
            cohere=cohere,
            separate=separate,
            match=match,
            speed=speed,
            vision=vision,
            separation=separation,
            color="green",
        )

        # Create black boids (the ones that flock)
        black_positions = self.rng.random(size=(black_population, 2)) * self.space.size
        black_directions = self.rng.uniform(-1, 1, size=(black_population, 2))
        Boid.create_agents(
            self,
            black_population,
            self.space,
            position=black_positions,
            direction=black_directions,
            cohere=cohere,
            separate=separate,
            match=match,
            speed=speed,
            vision=vision,
            separation=separation,
            color="black",
        )

        # For tracking statistics
        self.average_heading = None
        self.update_average_heading()

    def update_average_heading(self):
        """Calculate the average heading (direction) of all Boids."""
        if not self.agents:
            self.average_heading = 0
            return

        headings = np.array([agent.direction for agent in self.agents])
        mean_heading = np.mean(headings, axis=0)
        self.average_heading = np.arctan2(mean_heading[1], mean_heading[0])

    def step(self):
        """Run one step of the model.

        All agents are activated in random order using the AgentSet shuffle_do method.
        """
        self.agents.shuffle_do("step")
        self.update_average_heading()
