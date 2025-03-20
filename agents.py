"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior.
"""

import numpy as np
import random as rd 

from mesa.experimental.continuous_space import ContinuousSpaceAgent


class Boid(ContinuousSpaceAgent):
    """A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents
        - Separation: avoiding getting too close to any other agent
        - Alignment: trying to fly in the same direction as neighbors

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and direction (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        speed=1,
        direction=(1, 1),
        vision=1,
        separation=1,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        color="blue", 
        timeout=0, 
    ):
        """Create a new Boid flocker agent.

        Args:
            model: Model instance the agent belongs to
            speed: Distance to move per step
            direction: numpy vector for the Boid's direction of movement
            vision: Radius to look around for nearby Boids
            separation: Minimum distance to maintain from other Boids
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(space, model)
        self.position = position
        self.speed = speed
        self.direction = direction
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []
        self.color = color
        self.timeout = timeout

    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        if self.color == "black" or self.color == "green" or self.color == "red":
            # --- The original flocking logic ---
            neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
            self.neighbors = [n for n in neighbors if n is not self]
            self.blue_neighbors = [n for n in neighbors if (n is not self and n.color=="blue")]
            if len(self.blue_neighbors) > 0: 
                if self.color=="green":
                    self.timeout += 1
                    if self.timeout >= 10:
                        self.color="red"
                if self.color=="red":
                    self.timeout += 1
                    if self.timeout >= 35:
                        self.color="white" #removes the agent by turning it white!

            # If no neighbors, just move forward
            if not neighbors:
                self.position += self.direction * self.speed
                return

            delta = self.space.calculate_difference_vector(self.position, agents=neighbors)

            cohere_vector = delta.sum(axis=0) * self.cohere_factor
            separation_vector = (
                -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
            )
            match_vector = (
                np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
            )

            # Update direction based on the three behaviors
            self.direction += (cohere_vector + separation_vector + match_vector) / len(neighbors)
            # Normalize direction vector
            self.direction /= np.linalg.norm(self.direction)

            # Move boid
            self.position += self.direction * self.speed
        elif self.color=="white":
            self.color = rd.choice(["green", "black", "white"])  #randomly brings back, or "spawns" new players. 
            self.timeout=0
            self.position=self.model.rng.random(size=(2,)) * self.space.size
        else:
            # --- "Blue" boids: random wandering example ---
            # Generate a random direction
            random_dir = np.random.uniform(-1, 1, 2)
            # Normalize it
            random_dir /= np.linalg.norm(random_dir)
            # Assign to boid's direction
            self.direction = random_dir
            # Move
            self.position += self.direction * self.speed