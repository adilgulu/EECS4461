import os
import sys
import solara
import asyncio

sys.path.insert(0, os.path.abspath("../../../.."))

from mesa.examples.basic.boid_flockers.model import BoidFlockers
from mesa.visualization import Slider, SolaraViz, make_space_component


def boid_draw(agent):
    """
    Portrayal function:
      - For agents flagged for removal, return a dummy portrayal.
      - Blue cops remain fixed in size.
      - Regular players (black) and cheaters grow with their score.
      - Flagged cheaters (red) are drawn in red.
    """
    if getattr(agent, "to_remove", False):
        return {"color": "white", "size": 0, "marker": "o"}
    if agent.color == "blue":
        size = 20
        marker = "o"
    elif agent.color == "black":
        size = 20 + agent.score
        marker = "s"
    else:
        size = 20 + agent.score
        marker = (3, 0, 60)
    return {"color": agent.color, "size": size, "marker": marker}


@solara.component
def StatsPanel(model):
    """
    Stats panel that displays simulation statistics in organized groups.
    A timer forces periodic re-rendering.
    """
    tick, set_tick = solara.use_state(0)

    async def updater():
        while True:
            await asyncio.sleep(0.5)
            set_tick(lambda old: old + 1)

    def effect():
        task = asyncio.create_task(updater())
        return lambda: task.cancel()
    solara.use_effect(effect, [])

    # Gather stats from the model.
    stats = model.collect_stats()

    return solara.VBox([
        solara.Markdown("### Simulation Statistics"),
        solara.Div("", style={"height": "10px"}),

        solara.Markdown("#### Eliminated Agents"),
        solara.Text(f"Eliminated Regular Players: {stats['Eliminated Regular Players']}"),
        solara.Text(f"Eliminated Cheaters: {stats['Eliminated Cheaters']}"),
        solara.Div("", style={"height": "10px"}),

        solara.Markdown("#### Current Agents"),
        solara.Text(f"Current Regular Players: {stats['Current Regular Players']}"),
        solara.Text(f"Current Cheat Detectors: {stats['Current Cheat Detectors']}"),
        solara.Text(f"Current Cheaters: {stats['Current Cheaters']}"),
        solara.Text(f"Flagged Cheaters: {stats['Flagged Cheaters']}"),
        solara.Div("", style={"height": "10px"}),

        solara.Markdown("#### Score Statistics"),
        solara.Text(f"Regular Player Highest Score: {stats['Regular Player Highest Score']}"),
        solara.Text(f"Regular Player Lowest Score: {stats['Regular Player Lowest Score']}"),
        solara.Text(f"Cheat Bot Highest Score: {stats['Cheat Bot Highest Score']}"),
        solara.Text(f"Cheat Bot Lowest Score: {stats['Cheat Bot Lowest Score']}"),
        solara.Div("", style={"height": "10px"}),

        solara.Markdown("#### Detection Statistics (Current Step)"),
        solara.Text(f"True Positives: {stats['True Positives']}"),
        solara.Text(f"False Positives: {stats['False Positives']}"),
        solara.Text(f"True Negatives: {stats['True Negatives']}"),
        solara.Text(f"False Negatives: {stats['False Negatives']}"),
    ])


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "population_size": Slider(
        label="Number of boids",
        value=100,
        min=10,
        max=200,
        step=10,
    ),
    "width": 100,
    "height": 100,
    "speed": Slider(
        label="Speed of Boids",
        value=5,
        min=1,
        max=20,
        step=1,
    ),
    "vision": Slider(
        label="Vision of Bird (radius)",
        value=10,
        min=1,
        max=50,
        step=1,
    ),
    "separation": Slider(
        label="Minimum Separation",
        value=2,
        min=1,
        max=20,
        step=1,
    ),
}

# Create the simulation model.
model = BoidFlockers()

# Use built-in SolaraViz controls along with our StatsPanel.
page = SolaraViz(
    model,
    components=[
        make_space_component(agent_portrayal=boid_draw, backend="matplotlib"),
        StatsPanel,
    ],
    model_params=model_params,
    name="Boid Flocking Model",
)
