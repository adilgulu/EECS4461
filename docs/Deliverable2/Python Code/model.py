"""
Boids Flocking Model
===================
A Mesa implementation modified for a slither.io–like simulation.
Regular players (black) and cheaters (green) are removed when eliminated.
Cheaters flagged by cops (blue) are shown in red and later removed if still misbehaving.
New regular players and cheaters are added periodically.
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
        population_size=100,
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
        # Counters for eliminations.
        self.eliminated_regular_count = 0
        self.eliminated_cheater_count = 0

        # We'll store the step-based detection stats after each step:
        self.step_true_positives = 0
        self.step_false_positives = 0
        self.step_true_negatives = 0
        self.step_false_negatives = 0

        # Determine initial populations:
        blue_population = int(population_size * 0.16)  # 16% cops/detectors
        green_population = int(population_size * 0.05)   # 5% cheaters
        black_population = population_size - blue_population - green_population

        # --- Spawn cops/detectors (blue) on a grid ---
        grid_dim = int(np.ceil(np.sqrt(blue_population)))
        blue_positions = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                if len(blue_positions) < blue_population:
                    # Evenly space positions across the board.
                    x = (i + 0.5) * (width / grid_dim)
                    y = (j + 0.5) * (height / grid_dim)
                    blue_positions.append([x, y])
        blue_positions = np.array(blue_positions)
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
            color="blue",
        )

        # Create cheaters (green) at random positions.
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

        # Create regular players (black) at random positions.
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

        self.average_heading = None
        self.update_average_heading()

    def update_average_heading(self):
        """Calculate the average heading of all agents."""
        if not self.agents:
            self.average_heading = 0
            return
        headings = np.array([agent.direction for agent in self.agents])
        mean_heading = np.mean(headings, axis=0)
        self.average_heading = np.arctan2(mean_heading[1], mean_heading[0])

    def remove_dead_agents(self):
        """Remove agents flagged for removal and rebuild the space's agent dictionary."""
        dead_agents = [agent for agent in list(self.agents) if getattr(agent, "to_remove", False)]
        for agent in dead_agents:
            try:
                self.agents.remove(agent)
            except Exception:
                pass
        if hasattr(self.space, "_agent_points"):
            self.space._agent_points = {agent: agent.position for agent in self.agents}

    def add_new_agents(self):
        """Periodically add new regular players and cheaters."""
        # New regular player (black) with 50% chance per step.
        if self.random.random() < 0.50:
            pos = self.rng.random(size=(1, 2)) * self.space.size
            direc = self.rng.uniform(-1, 1, size=(1, 2))
            Boid.create_agents(
                self,
                1,
                self.space,
                position=pos,
                direction=direc,
                cohere=0.03,
                separate=0.015,
                match=0.05,
                speed=1,
                vision=10,
                separation=2,
                color="black",
            )
        # New cheater (green) with 15% chance per step.
        if self.random.random() < 0.15:
            pos = self.rng.random(size=(1, 2)) * self.space.size
            direc = self.rng.uniform(-1, 1, size=(1, 2))
            Boid.create_agents(
                self,
                1,
                self.space,
                position=pos,
                direction=direc,
                cohere=0.03,
                separate=0.015,
                match=0.05,
                speed=1,
                vision=10,
                separation=2,
                color="green",
            )

    def compute_detection_stats(self):
        """
        Classify each agent in the current state:
          - True Positive (TP):  flagged & actually a cheater
          - False Positive (FP): flagged & not a cheater
          - False Negative (FN): not flagged & is a cheater
          - True Negative (TN):  not flagged & not a cheater
        """
        tp = fp = fn = tn = 0
        for agent in self.agents:
            if agent.initial_color == "green":
                # Cheater
                if agent.flagged:
                    tp += 1
                else:
                    fn += 1
            else:
                # Not a cheater
                if agent.flagged:
                    fp += 1
                else:
                    tn += 1

        self.step_true_positives = tp
        self.step_false_positives = fp
        self.step_false_negatives = fn
        self.step_true_negatives = tn

    def collect_stats(self):
        """Collect simulation statistics for display in the UI."""
        stats = {}
        stats["Eliminated Regular Players"] = self.eliminated_regular_count
        stats["Eliminated Cheaters"] = self.eliminated_cheater_count

        # Current agents by type.
        current_regular = [a for a in self.agents if getattr(a, "initial_color", a.color) == "black"]
        current_cheat_detectors = [a for a in self.agents if getattr(a, "initial_color", a.color) == "blue"]
        current_cheaters = [a for a in self.agents if getattr(a, "initial_color", a.color) == "green"]

        stats["Current Regular Players"] = len(current_regular)
        stats["Current Cheat Detectors"] = len(current_cheat_detectors)
        stats["Current Cheaters"] = len(current_cheaters)

        flagged_cheaters = [
            a for a in self.agents
            if a.initial_color == "green" and a.color == "red"
        ]
        stats["Flagged Cheaters"] = len(flagged_cheaters)

        # Score statistics.
        regular_scores = [a.score for a in current_regular]
        if regular_scores:
            stats["Regular Player Highest Score"] = round(max(regular_scores), 2)
            stats["Regular Player Lowest Score"] = round(min(regular_scores), 2)
        else:
            stats["Regular Player Highest Score"] = stats["Regular Player Lowest Score"] = 0

        cheater_scores = [a.score for a in current_cheaters]
        if cheater_scores:
            stats["Cheat Bot Highest Score"] = round(max(cheater_scores), 2)
            stats["Cheat Bot Lowest Score"] = round(min(cheater_scores), 2)
        else:
            stats["Cheat Bot Highest Score"] = stats["Cheat Bot Lowest Score"] = 0

        # Detection statistics (based on current state)
        stats["True Positives"] = self.step_true_positives
        stats["False Positives"] = self.step_false_positives
        stats["False Negatives"] = self.step_false_negatives
        stats["True Negatives"] = self.step_true_negatives

        return stats

    def step(self):
        """Run one simulation step."""
        self.agents.shuffle_do("step")   # Agents move and possibly flag others
        self.remove_dead_agents()         # Remove those marked for removal
        self.add_new_agents()             # Add new agents (regular or cheaters)
        self.update_average_heading()     # Recalculate average heading

        # Now compute detection stats based on final state after this step
        self.compute_detection_stats()
