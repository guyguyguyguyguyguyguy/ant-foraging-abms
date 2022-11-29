import random
from typing import Dict, List, Optional, Union

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation

import antMethodsClasses
from antClasses import Ant, Forager, Nestmate
from dataCollection import *
from dataCollection import modDataCollector
from helperFunctions import *
from inspectClass import Inspectable
from validatorClass import *


# Model 'abstract' class
class BaseModel(Model, Inspectable):
    """
    class
    """

    colony_satiation_level = 0.99
    model_step = 0
    nestmates_pos_dict: Dict[List[float], Nestmate] = {}

    def __init__(
        self,
        nest_depth: int,
        exit_size: int,
        step_sizes: List[float],
        motion_threshold: float,
        trophallaxis_method: antMethodsClasses.Trophallaxis,
        movement_method: antMethodsClasses.Movement,
        propagate_food: bool,
        number_of_ants: Optional[Dict[str, Union[int, str]]],
        nest_height: int,
        start_in_nest: bool,
        nestmate_capacity: float,
        max_steps: int,
        deployment: Optional[Dict[str, object]],
        forager_proportion_to_give: float,
        nestmate_proportion_to_give: float,
        homogenise: bool,
        nestmate_movement: bool,
        forager_lag: int,
        diff_test: bool,
        full_nest_ants: int,
        forager_interaction_rate: float,
        repeat: int,
        verbose: bool,
        space_test: bool,
        nestmate_bias: bool,
        inert_nestants: bool,
    ) -> None:
        super().__init__()

        self.schedule = RandomActivation(self)
        self.step_sizes = step_sizes
        self.nest_depth = nest_depth
        self.nest_height = nest_height
        self.repeat = repeat
        self.verbose = verbose
        self.space_test = space_test
        self.nestmate_bias = nestmate_bias
        self.inert_nestants = inert_nestants

        # todo: maybe change exit to a list of exit cells
        self.exit_size = exit_size
        self.length = self.nest_depth + self.exit_size
        # if want different trophallaxis for different agents, need to change the way this is done,
        # maybe a dictionary of trophallaxis options?
        self.trophallaxis_method = trophallaxis_method
        self.movement_method = movement_method
        self.motion_threshold = motion_threshold
        self.propagate_food = propagate_food
        self.nestmate_capacity = nestmate_capacity
        self.start_in_nest = start_in_nest
        self._deployment = deployment
        if number_of_ants is None:
            self._number_of_ants = {"Nestmate": "all", "Forager": "all"}
        else:
            self._number_of_ants = number_of_ants

        if deployment is None:
            self._deployment = {"Nestmate": None, "Forager": None}
        else:
            self._deployment = deployment

        self.max_steps = max_steps
        self.homogenise = homogenise
        self.nestmate_movement = nestmate_movement
        self.forager_lag = forager_lag
        self.diff_test = diff_test
        if self.diff_test:
            self.full_nest_ants = full_nest_ants

        self.forager_interaction_rate = forager_interaction_rate

        if start_in_nest:
            self.all_possible_positions = {
                "Nestmate": [
                    [x, y]
                    for y in range(self.nest_height)
                    for x in range(self.exit_size, self.length)
                ],
                "Forager": [
                    [x, y]
                    for x in range(0, self.length)
                    for y in range(self.nest_height)
                ],
            }
        else:
            self.all_possible_positions = {
                "Nestmate": [
                    [x, y]
                    for x in range(self.exit_size, self.length)
                    for y in range(self.nest_height)
                ],
                "Forager": [[0, 0] for y in range(self._number_of_ants["Forager"])],
            }

        self.forager_proportion_to_give = forager_proportion_to_give
        self.nestmate_proportion_to_give = nestmate_proportion_to_give

        self.step_data_collector = None
        self.visit_data_collector = None
        self.interaction_data_collector = None
        self.forager_data_collector = None

        self.agents = []
        self.foragers = []
        self.nestmates = []
        self.initialize_step_data_collector()
        self.initialize_visit_data_collector()
        self.initialize_interaction_data_collector()
        self.initialize_forager_data_collector()

        validator = OneDValidation(self)
        # Validation of a number of ants and deployment
        self.number_of_ants, self.deployment = validator.set_values(
            self._number_of_ants, self._deployment
        )

        if start_in_nest:
            self.all_possible_positions = {
                "Nestmate": [
                    [x, y]
                    for y in range(self.nest_height)
                    for x in range(self.exit_size, self.length)
                ],
                "Forager": [
                    [x, y]
                    for x in range(0, self.length)
                    for y in range(self.nest_height)
                ],
            }
        else:
            self.all_possible_positions = {
                "Nestmate": [
                    [x, y]
                    for x in range(self.exit_size, self.length)
                    for y in range(self.nest_height)
                ],
                "Forager": [[0, 0] for y in range(self.number_of_ants["Forager"])],
            }

    # Instantiate all ants in the model with their required attributes. Attribute values for each class are taken model attributes defined by the user (using get_properties method)
    def populate(self) -> None:
        subclasses = [x.__name__ for x in Ant.__subclasses__()]
        for subclass in subclasses:
            if self.number_of_ants[subclass]:  # so, not None
                properties = get_properties(self, globals()[subclass])  # your hf method
                positions = self.get_positions(
                    subclass
                )  # this is a method which generates/gets
                # the positions.
                if positions:
                    for position in positions:
                        self.create_agent(
                            subclass, position, properties
                        )  # id is resolved in within the Ant class

    # Instantiate agent, add it to model list of agents and scheduling
    def create_agent(
        self, subclass, position: List[int], properties: List[object]
    ) -> None:
        subclass = globals()[subclass]
        ant = subclass(self, position, *properties)  # id resolved in constructor
        ant.add()
        # ant.place()

    # Default values for ant positions
    def get_positions(self, subclass) -> Optional[List[List[int]]]:

        if self.number_of_ants[subclass] is None:
            if subclass == "Nestmate":
                # positions = self.calculate_all_grid_positions(subclass)
                return self.all_possible_positions[subclass]

            elif subclass == "Forager":
                positions = [[0, 0]]
                return positions

        elif self.number_of_ants[subclass] in ["all", "All"]:
            positions = self.calculate_all_grid_positions(subclass)
            return positions

        elif self.number_of_ants[subclass] == 0:
            positions = None
            return positions

        elif 0 < self.number_of_ants[subclass] <= (self.nest_depth * self.nest_height):
            positions = self.calculate_grid_positions(subclass)
            return positions

    @abstractmethod
    def calculate_grid_positions(self, subclass) -> List[List[int]]:
        pass

    @abstractmethod
    def calculate_all_grid_positions(self, subclass) -> List[List[int]]:
        pass

    def proportion_of_nest(self, subclass) -> int:
        return int(
            self.number_of_ants[subclass] * len(self.all_possible_positions[subclass])
        )

    # region Setting Data Collection
    @abstractmethod
    def set_step_model_reporters(self) -> Dict:
        pass

    @abstractmethod
    def set_step_agent_reporters(self) -> Dict:
        pass

    # Data collectors to store information and output dataframe of various model and ant attributes

    # Data collected at each step for all ants
    def initialize_step_data_collector(self) -> None:
        model_reporters = self.set_step_model_reporters()
        agent_reporters = self.set_step_agent_reporters()
        self.step_data_collector = modDataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

    @abstractmethod
    def set_visit_model_reporters(self) -> Dict:
        pass

    @abstractmethod
    def set_visit_agent_reporters(self) -> Dict:
        pass

    # Data collected at each visit
    def initialize_visit_data_collector(self) -> None:
        model_reporters = self.set_visit_model_reporters()
        agent_reporters = self.set_visit_agent_reporters()
        self.visit_data_collector = modDataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

    @abstractmethod
    def set_interaction_model_reporters(self) -> Dict:
        pass

    @abstractmethod
    def set_interaction_agent_reporters(self) -> Dict:
        pass

    # Data collected at each interaction
    def initialize_interaction_data_collector(self) -> None:
        model_reporters = self.set_interaction_model_reporters()
        agent_reporters = self.set_interaction_agent_reporters()
        self.interaction_data_collector = modDataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

    @abstractmethod
    def set_forager_model_reporters(self) -> Dict:
        pass

    @abstractmethod
    def set_forager_agent_reporters(self) -> Dict:
        pass

    # Data collected at each step only for forager
    def initialize_forager_data_collector(self) -> None:
        model_reporters = self.set_forager_model_reporters()
        agent_reporters = self.set_forager_agent_reporters()
        self.forager_data_collector = modDataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

    def run(self) -> None:
        self.nestmate_number = len(self.nestmates)
        while not self.is_done():
            self.step()

    def step(self) -> None:
        # If homogenise, at every forager exit, all ants crops become the average
        if self.homogenise and get_foragers_at_exit(self):
            colony_state = get_colony_state(self)
            for i in get_nest_ants(self, Nestmate):
                i.crop_state = colony_state

        # Randomly mix nestmates at every forager exit
        if self.space_test and get_foragers_at_exit(self):
            space_model(self, Nestmate)

        # If nestmates can move, then do so every time the forager exits
        if get_foragers_at_exit(self) and self.model_step > 1:
            if self.nestmate_movement == "full":
                nestmate_movement(self, Nestmate)
            elif self.nestmate_movement >= 1:
                nestmate_movement_limited(
                    self, Nestmate, 1.1, self.nestmate_movement, bias=self.nestmate_bias
                )

        # Collect data
        self.step_data_collector.collect(self)
        self.forager_data_collector.collect(self, forag=True)
        if get_foragers_at_exit(self):
            self.visit_data_collector.collect(self, forag=True)

        # Print colony state for last 5 steps of model if user requests verbose
        if (self.max_steps - self.model_step) <= 5 and self.verbose:
            print(get_colony_state(self))

        # maybe want this to work for nestmate interactions too, or maybe want a new dataframe
        if get_interacting_forager(self.foragers):
            self.interaction_data_collector.collect(self, forag=True)

        self.nestmates_pos_dict = {tuple(n.position): n for n in self.nestmates}
        self.model_step += 1
        if self.inert_nestants:
            for agent in self.agents:
                agent.agent_step += 1
        self.schedule.step()

    def is_done(self) -> Optional[bool]:
        # todo: fix this, currently doesnt work as when creating average step dataframe, the differeing number of
        # steps between runs causes problems, makes the dataframe huge as cannot add results of each run

        # if get_colony_state(self) > self.colony_satiation_level:
        #     print("\t \t Colony satiated")
        #     Ant._id = 0
        #     return True

        if self.model_step == self.max_steps:
            if self.verbose:
                print(
                    "\t \t Maximum number of steps taken, finishing condition was not met"
                )
            Ant._id = 0
            return True


# One dimensional model class
class OneDModel(BaseModel):
    """
    class
    """

    name = "one_d_model"

    def __init__(
        self,
        nest_depth,
        exit_size,
        step_sizes,
        motion_threshold,
        trophallaxis_method,
        movement_method,
        propagate_food=0,
        number_of_ants=None,
        nest_height=1,
        start_in_nest=False,
        nestmate_capacity=1,
        max_steps=10000,
        deployment=None,
        forager_proportion_to_give=0.15,
        nestmate_proportion_to_give=0.15,
        homogenise=False,
        nestmate_movement=False,
        forager_lag=0,
        diff_test=False,
        full_nest_ants=0,
        forager_interaction_rate=1,
        repeat=1,
        verbose=False,
        space_test=False,
        nestmate_bias=False,
        inert_nestants=True,
    ):
        super().__init__(
            nest_depth,
            exit_size,
            step_sizes,
            motion_threshold,
            trophallaxis_method,
            movement_method,
            propagate_food,
            number_of_ants,
            nest_height,
            start_in_nest,
            nestmate_capacity,
            max_steps,
            deployment,
            forager_proportion_to_give,
            nestmate_proportion_to_give,
            homogenise,
            nestmate_movement,
            forager_lag,
            diff_test,
            full_nest_ants,
            forager_interaction_rate,
            repeat,
            verbose,
            space_test,
            nestmate_bias,
            inert_nestants,
        )

        # Not used, mesa function
        self.grid = MultiGrid(exit_size + nest_depth, 1, False)

        self.entrance = [0, 0]

    # Get positions of ants based on user input
    def calculate_grid_positions(self, subclass) -> Optional[List[List[int]]]:
        if isinstance(self.deployment[subclass], list):
            self.number_of_ants[subclass] = len(self.deployment[subclass])
            return self.deployment[subclass]

        elif 0 < self.number_of_ants[subclass] < 1:

            if self.deployment[subclass] is "Random":
                positions = random.sample(
                    self.all_possible_positions[subclass],
                    int(
                        self.number_of_ants[subclass]
                        * len(self.all_possible_positions[subclass])
                    ),
                )
                self.number_of_ants[subclass] = len(positions)
                return positions

            else:
                positions = [
                    self.all_possible_positions[subclass][x]
                    for x in range(self.proportion_of_nest(subclass))
                ]
                self.number_of_ants[subclass] = len(positions)
                return positions

        elif self.deployment[subclass] is None:
            positions = [
                self.all_possible_positions[subclass][x]
                for x in range(self.number_of_ants[subclass])
            ]
            self.number_of_ants[subclass] = len(positions)
            return positions

    def calculate_all_grid_positions(self, subclass) -> List[List[int]]:
        positions = self.all_possible_positions[subclass]
        self.number_of_ants[subclass] = len(positions)
        return positions

    # Reporters for data collectors
    def set_step_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_step_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "crop": "crop_state",
            "id": "unique_id",
            "position": lambda a: a.position.copy()[0],
        }

    def set_interaction_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_interaction_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "forager crop": lambda a: a.crop_state,
            "Nest ant crop": lambda a: get_recipients(a).crop_state,
            "interaction volume": lambda a: a.food_given,
            "position": lambda a: a.position[0],
            "trip": lambda a: a.trip,
        }

    def set_visit_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_visit_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "exiting crop": lambda a: a.exiting_crop,
            "trip length": lambda a: a.trip_length,
        }

    def set_forager_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_forager_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "crop": lambda a: a.crop_state,
            "trip": lambda a: a.trip,
            "position": lambda a: a.position.copy()[0],
            "exiting crop": lambda a: a.exiting_crop,
            "le crop": lambda a: a.crop_at_le,
            "fall thresh": lambda a: a.crop_below_thresh,
        }


# Two dimensional model
class TwoDModel(BaseModel):
    """
    class
    """

    name = "two_d_model"

    def __init__(
        self,
        nest_depth,
        exit_size,
        step_sizes,
        motion_threshold,
        trophallaxis_method,
        movement_method,
        propagate_food=0,
        number_of_ants=None,
        nest_height=1,
        start_in_nest=False,
        nestmate_capacity=1,
        max_steps=10000,
        deployment=None,
        forager_proportion_to_give=0.15,
        nestmate_proportion_to_give=0.15,
        homogenise=False,
        nestmate_movement=False,
        forager_lag=0,
        diff_test=False,
        full_nest_ants=0,
        forager_interaction_rate=1,
        repeat=1,
        verbose=False,
        space_test=False,
        nestmate_bias=False,
        inert_nestants=True,
    ):
        super().__init__(
            nest_depth,
            exit_size,
            step_sizes,
            motion_threshold,
            trophallaxis_method,
            movement_method,
            propagate_food,
            number_of_ants,
            nest_height,
            start_in_nest,
            nestmate_capacity,
            max_steps,
            deployment,
            forager_proportion_to_give,
            nestmate_proportion_to_give,
            homogenise,
            nestmate_movement,
            forager_lag,
            diff_test,
            full_nest_ants,
            forager_interaction_rate,
            repeat,
            verbose,
            space_test,
            nestmate_bias,
            inert_nestants,
        )

        self.entrance = [0, 0]
        self.grid = MultiGrid(exit_size + nest_depth, self.nest_height, False)

    def calculate_grid_positions(self, subclass) -> Optional[List[List[int]]]:
        if isinstance(self.deployment[subclass], list):
            self.number_of_ants[subclass] = len(self.deployment[subclass])
            return self.deployment[subclass]

        elif 0 < self.number_of_ants[subclass] < 1:

            if self.deployment[subclass] is "Random":
                positions = random.sample(
                    self.all_possible_positions[subclass],
                    int(
                        self.number_of_ants[subclass]
                        * len(self.all_possible_positions[subclass])
                    ),
                )
                self.number_of_ants[subclass] = len(positions)
                return positions

            else:
                positions = [
                    self.all_possible_positions[subclass][x]
                    for x in range(self.proportion_of_nest(subclass))
                ]
                self.number_of_ants[subclass] = len(positions)
                return positions

        elif self.deployment[subclass] is None:
            positions = [
                self.all_possible_positions[subclass][x]
                for x in range(self.number_of_ants[subclass])
            ]
            self.number_of_ants[subclass] = len(positions)
            return positions

    def calculate_all_grid_positions(self, subclass) -> List[List[int]]:
        positions = self.all_possible_positions[subclass]
        self.number_of_ants[subclass] = len(positions)
        return positions

    def set_step_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_step_agent_reporters(self) -> Dict:
        if self.repeat == 1:
            return {
                "step": "agent_step",
                "crop": "crop_state",
                "id": "unique_id",
                "position": lambda a: a.pos.copy(),
                "trip": lambda a: a.trip if isinstance(a, Forager) else None,
            }
        return {
            "step": "agent_step",
            "crop": "crop_state",
            "id": "unique_id",
            "trip": lambda a: a.trip if isinstance(a, Forager) else None,
        }

    def set_interaction_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_interaction_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "forager crop": lambda a: a.crop_state,
            "Nest ant crop": lambda a: get_recipients(a).crop_state,
            "interaction volume": lambda a: a.food_given,
            "position": lambda a: a.position,
            "depth": lambda a: np.linalg.norm(a.position),
            "trip": lambda a: a.trip,
        }

    def set_visit_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_visit_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "exiting crop": lambda a: a.exiting_crop
            if isinstance(a, Forager)
            else None,
            "trip length": lambda a: a.trip_length if isinstance(a, Forager) else None,
        }

    def set_forager_model_reporters(self) -> Dict:
        return {
            "step": "model_step",
            "colony state": lambda m: get_colony_state(m),
            "empty colony state": lambda m: get_empty_colony_state(m),
            "repeat": "repeat",
        }

    def set_forager_agent_reporters(self) -> Dict:
        return {
            "step": "agent_step",
            "crop": lambda a: a.crop_state if isinstance(a, Forager) else None,
            "trip": lambda a: a.trip if isinstance(a, Forager) else None,
            "position": lambda a: a.position.copy() if isinstance(a, Forager) else None,
            "depth": lambda a: np.linalg.norm(a.position),
            "exiting crop": lambda a: a.exiting_crop
            if isinstance(a, Forager)
            else None,
            "le crop": lambda a: a.crop_at_le if isinstance(a, Forager) else None,
        }
