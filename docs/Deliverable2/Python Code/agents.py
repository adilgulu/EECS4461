"""
A Boid (bird-oid) agent for a slither.io–like simulation.
Uses numpy arrays to represent vectors for efficient movement and interaction.
"""

import numpy as np
import random as rd

from mesa.experimental.continuous_space import ContinuousSpaceAgent


class Boid(ContinuousSpaceAgent):
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
        in_burst=False,
        burst_timer=0,
        score=0,
        growth_rate=0,
    ):
        """Initialize the agent and record its original type via initial_color."""
        super().__init__(space, model)
        self.initial_color = color  # "blue" for cop, "green" for cheater, "black" for regular player.
        self.color = color
        self.position = np.array(position, dtype=float)
        self.speed = speed
        self.direction = np.array(direction, dtype=float)
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []
        self.timeout = timeout
        self.in_burst = in_burst
        self.burst_timer = burst_timer
        self.score = score
        self.prev_score = score  # Store previous score for growth rate calculation.
        self.to_remove = False
        self.flagged = False
        self.flagged_duration = 0  # Number of consecutive steps flagged
        self.growth_rate = growth_rate

    def step(self):
        old_score = self.score

        # --- Pellet Consumption & Survival ---
        # Only non-blue agents (regular and cheaters) gain score.
        self.score += 0.1  # Constant survival increment
        if rd.random() < 0.1:
            pellet_score = rd.random() * 6 / 10
            if self.color != "blue":
                self.score += pellet_score

        # --- Behavior for Regular Players (black) ---
        if self.color == "black":
            neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
            self.neighbors = [n for n in neighbors if n is not self]
            collision_distance = 0.3  # adjust as needed
            for n, d in zip(neighbors, distances):
                if d < collision_distance and n.score > self.score:
                    # Regular player eliminated on collision with a higher-scoring agent.
                    self.to_remove = True
                    self.model.eliminated_regular_count += 1
                    return  # Terminate step; no further updates.
            if not neighbors:
                self.position += self.direction * self.speed
                self.growth_rate = self.score - old_score
                self.prev_score = self.score
                return
            delta = self.space.calculate_difference_vector(self.position, agents=neighbors)
            cohere_vector = delta.sum(axis=0) * self.cohere_factor
            separation_vector = (
                -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
            )
            match_vector = (
                np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
            )
            self.direction += (cohere_vector + separation_vector + match_vector) / len(neighbors)
            norm = np.linalg.norm(self.direction)
            if norm:
                self.direction /= norm
            self.position += self.direction * self.speed

        # --- Behavior for Cheaters (green) or Flagged Cheaters (red) ---
        elif self.color in ("green", "red"):
            neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
            self.neighbors = [n for n in neighbors if n is not self]
            if not neighbors:
                self.position += self.direction * self.speed
                self.growth_rate = self.score - old_score
                self.prev_score = self.score
                return
            delta = self.space.calculate_difference_vector(self.position, agents=neighbors)
            cohere_vector = delta.sum(axis=0) * self.cohere_factor
            separation_vector = (
                -1 * delta[distances < self.separation].sum(axis=0) * self.separate_factor
            )
            match_vector = (
                np.asarray([n.direction for n in neighbors]).sum(axis=0) * self.match_factor
            )
            self.direction += (cohere_vector + separation_vector + match_vector) / len(neighbors)
            norm = np.linalg.norm(self.direction)
            if norm:
                self.direction /= norm
            self.position += self.direction * self.speed
            # Cheat bots get a bonus to simulate their unfair advantage.
            self.score += 0.5
            if not self.flagged:
                self.flagged_duration = 0

        # --- Behavior for Cops/Detectors (blue) ---
        elif self.color == "blue":
            growth_threshold = 0.6         # Rate of score increase that is considered "odd"
            flag_duration_threshold = 5    # Steps to wait before removal
            neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
            min_flagging_score = 20
            for n in neighbors:
                if n.growth_rate > growth_threshold and n.score > min_flagging_score:
                    # If they're suspicious, flag them.
                    if not n.flagged:
                        n.flagged = True
                        n.flagged_duration = 1
                        n.color = "red"  # Show flagged color
                    else:
                        # Only increment if not already marked for removal.
                        if not n.to_remove:
                            n.flagged_duration += 1
                            if n.flagged_duration >= flag_duration_threshold:
                                if n.initial_color == "black":
                                    self.model.eliminated_regular_count += 1
                                elif n.initial_color == "green":
                                    self.model.eliminated_cheater_count += 1
                                n.to_remove = True

            # Random wandering for cops.
            random_dir = np.random.uniform(-1, 1, 2)
            norm = np.linalg.norm(random_dir)
            if norm:
                random_dir /= norm
            self.direction = random_dir
            self.position += self.direction * self.speed

        # --- Update Growth Rate ---
        if not self.to_remove:
            self.growth_rate = self.score - old_score
            self.prev_score = self.score
