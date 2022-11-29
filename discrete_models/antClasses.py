from __future__ import annotations

import copy
import random
from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
from mesa import Agent

import antMethodsClasses
import helperFunctions as hf
from inspectClass import Inspectable


# 'Abstract' ant class
class Ant(Agent, Inspectable):
    """Abstract ant class

    Parameters
    ---------
    _id : int
        ant unique id
    agent_step : int
        current step the agent is on in the simulation
    food_give : float
        amount of food given in last trophallaxis interaction
    count_trip_length : int
        current trip length of the ant
    interaction : bool
        bool whether the ant has interacted on this step
    interacting_neighbour : Ant, optional
        trophallaxis interaction recipient ant if there is one
    trip_length : int
        length of last trip
    pos : List[float]
        current ant position (continuous)
    position : List[float]
        current ant position (discrete)

    Methods
    ------
    add(self)
        add ant to model, dependeant on ant type
    place(self)
        place agent on model grid (redundant)
    """

    _id: int = 0
    agent_step: int = 0
    food_given: float = 0
    count_trip_length: int = 0
    interaction: bool = False
    interacting_neighbour: Union["Ant", bool, None] = None
    trip_length: Optional[int] = None
    pos: Optional[List[float]] = None

    # Ant id is incremented by one on each instantiation of a class that inherits from Ant
    def __init__(self, model: object, position: List[float]):
        super().__init__(Ant._id, model)
        Ant._id += 1
        self._position = position
        self.pos = [int(x) for x in position]

    @property
    def position(self) -> List[float]:
        return self._position

    # pos required by mesa module, however, it is not used anymore
    @position.setter
    def position(self, value: List[float]):
        self._position = value
        self.pos = [int(x) for x in self.position]

    @abstractmethod
    def get_interaction_neighbour(self) -> None:
        pass

    # Add agent to model agents and scheduling for abm
    def add(self) -> None:
        """Add ant to model, dependant on ant type."""
        if isinstance(self, Forager):
            self.model.foragers.append(self)
        else:
            self.model.nestmates.append(self)
        if self.model.inert_nestants and isinstance(self, Nestmate):
            self.model.agents.append(self)
        else:
            self.model.agents.append(self)
            self.model.schedule.add(self)

    def place(self) -> None:
        """Redundant."""
        print(self.position)
        self.model.grid.place_agent(self, self.position)

    @abstractmethod
    def step(self) -> None:
        pass


class Forager(Ant):
    """Forager ant class.

    Forager ants are those that leave the nest to collect food, store it in their crop, and then move through the nest distributing this food.

    Parameters
    ---------
    crop_state : float
        proportion of crop filled [0 - 1]
    exiting_crop : float, optional

    """

    crop_state: float = 1
    exiting_crop: Optional[float] = 0
    trip: int = 0
    lag_counter: int = 0
    at_leading_edge: bool = False
    crop_at_le: Optional[float] = None
    crop_below_thresh: bool = False
    previous_crop: float = 0

    def __init__(
        self,
        model: object,
        position: List[float],
        motion_threshold: float,
        step_sizes: float,
        trophallaxis_method: antMethodsClasses.Trophallaxis,
        movement_method: antMethodsClasses.Movement,
        forager_proportion_to_give: float,
    ) -> None:
        super().__init__(model, position)

        self.threshold = motion_threshold
        self.step_sizes = step_sizes
        self.trophallaxis_method = trophallaxis_method
        self.movement_method = movement_method
        self.proportion_to_give = forager_proportion_to_give
        self.interaction_rate = self.model.forager_interaction_rate

    def trophallaxis(self) -> None:
        self.trophallaxis_method.trophallaxis(self)

    def move(self) -> None:
        self.movement_method.move(self, self.model)

    # Choose one other ant on same cell for forager to interact with
    def get_interaction_neighbour(self) -> None:
        # interacting_neighbour = list(filter((lambda x: x.position == agent.pos and x.unique_id !=
        #                                                agent.unique_id),neighbours))

        if self.pos != [0, 0]:
            interacting_neighbour = self.model.nestmates_pos_dict.get(
                (int(self.pos[0]), int(self.pos[1]))
            )
            self.interacting_neighbour = interacting_neighbour

        else:
            self.interacting_neighbour = False

    def crop_drop(self) -> None:
        if self.crop_state <= self.threshold and self.previous_crop > self.threshold:
            self.crop_below_thresh = True
        else:
            self.crop_below_thresh = False

    def step(self) -> None:
        self.previous_crop = copy.copy(self.crop_state)
        if self.model.diff_test:
            self.agent_step = self.model.model_step
            self.move()
            pass

        else:
            self.agent_step = self.model.model_step

            if self.position == [0, 0]:
                self.trip_length = self.count_trip_length
                # print('waiting %s, position %s' %(self.lag_counter, self.position))
                if self.agent_step == 1 or self.lag_counter >= self.model.forager_lag:
                    self.crop_state = 1
                    self.count_trip_length = 0
                    self.trip += 1
                    self.move()
                    self.get_interaction_neighbour()
                    if (
                        (not self.at_leading_edge)
                        and self.interacting_neighbour
                        and (self.interacting_neighbour.crop_state < 0.9)
                    ):
                        self.crop_at_le = self.crop_state
                        self.at_leading_edge = True
                    else:
                        self.crop_at_le = None
                    if hf.decision_making(self.interaction_rate):
                        self.trophallaxis()
                    self.lag_counter = 1
                    self.exiting_crop = None
                    # print('Done waiting')
                else:
                    self.lag_counter += 1

            elif self.position != [0, 0]:
                self.move()
                self.get_interaction_neighbour()
                if (
                    (not self.at_leading_edge)
                    and self.interacting_neighbour
                    and (self.interacting_neighbour.crop_state < 0.85)
                ):
                    self.crop_at_le = self.crop_state
                    self.at_leading_edge = True
                else:
                    self.crop_at_le = None
                if hf.decision_making(self.interaction_rate):
                    self.trophallaxis()
                self.count_trip_length += 1
                if self.position == [0, 0]:
                    self.exiting_crop = self.crop_state
                    self.trip_length = self.count_trip_length
                    self.at_leading_edge = False

                else:
                    self.exiting_crop = None
                # self.trip_length = None

        self.crop_drop()


class Nestmate(Ant):
    """
    class
    """

    crop_state: float = 0
    interaction_rate: Optional[float] = None
    position_selector = None

    def __init__(
        self,
        model: object,
        position: List[float],
        nestmate_capacity: float,
        trophallaxis_method: antMethodsClasses.Trophallaxis,
        nestmate_proportion_to_give: float,
    ) -> None:
        super().__init__(model, position)

        self.capacity = nestmate_capacity
        self.trophallaxis_method = trophallaxis_method
        self.proportion_to_give = nestmate_proportion_to_give

        if self.model.diff_test:
            if self.unique_id in range(1, self.model.full_nest_ants + 1):
                self.crop_state = 1

    def get_interaction_neighbour(self) -> None:
        interacting_neighbours = [
            x
            for x in self.model.agents
            if abs(sum(x.position) - sum(self.position)) <= 1
            and x is not self
            and isinstance(x, Nestmate)
        ]
        interacting_neighbour = random.choice(interacting_neighbours)

        if interacting_neighbour:
            self.interacting_neighbour = interacting_neighbour

        else:
            self.interacting_neighbour = False

    def get_interaction_rate(self) -> None:
        a_rate = self.model.propagate_food
        """
        Something, maybe colony state dependent, maybe space or time dependent
        """
        self.interaction_rate = a_rate

    def trophallaxis(self) -> None:
        if hf.decision_making(self.interaction_rate):
            self.trophallaxis_method.trophallaxis(self)

    def step(self) -> None:
        self.agent_step = self.model.model_step
        if self.model.propagate_food > 0 and self.crop_state > 0:
            self.get_interaction_neighbour()
            self.get_interaction_rate()
            self.trophallaxis()
        else:
            pass
