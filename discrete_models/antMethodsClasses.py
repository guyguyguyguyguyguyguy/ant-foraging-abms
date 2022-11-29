import copy
import random
from abc import ABC, abstractmethod
from operator import add

import numpy as np

from antClasses import Nestmate


class MovementCreator:
    """Class to select correct forager movement class based on model paramters

    Attributes
    ---------
    bias : List[float]
        list of floats representing the movement probabilities below and above threshold
    f_inertial_force : float
        float representing the bias of the forager to go inward if it has just entered the nest
    b_inertial_force : float
        float representing the bias of the forager to go outward if it has reached the end of the nest
    inertia_weight :
        strength of forwad and backward inertia
    stochastic : bool
        bool to choose for deterministic or stochastic movement in 1D
    order_test : bool
        bool to choose for an order test
    two_d : bool
        bool to choose a 2D nest
    extreme_move : bool (not used)
        bool to choose extreme movement
    aver_veloc : bool
        use average velocity for movement
    step_size : List[float]
        forager step size for average velocity movement rule

    Methods
    ------
    __new__(args)
        returns the correct movement class based on parameters
    """

    def __init__(
        self,
        bias=None,
        f_inertial_force=0,
        b_inertial_force=0,
        inertia_weight=0,
        stochastic=True,
        order_test=False,
        two_d=False,
        extreme_move=False,
        aver_veloc=False,
        step_size=None,
    ):
        self.bias = bias
        self.f_inertial_force = f_inertial_force
        self.b_inertial_force = b_inertial_force
        self.inertia_weight = inertia_weight
        self.stochastic = stochastic
        self.order_test = order_test
        self.two_d = two_d
        self.extreme_move = extreme_move
        self.aver_veloc = aver_veloc
        self.step_size = step_size

    def __new__(
        cls,
        bias,
        f_inertial_force,
        b_inertial_force,
        inertia_weight,
        stochastic,
        order_test,
        two_d,
        extreme_move,
        aver_veloc,
        step_size,
    ):
        if order_test:
            return SpacelessMovement()

        elif aver_veloc:
            return AverageVelocity(step_size=step_size)

        # elif extreme_move:
        #     return ExtremeMovement(bias = bias)

        elif two_d:
            return TDStochasticMovement(
                bias=bias,
                f_inertia_force=f_inertial_force,
                b_inertia_force=b_inertial_force,
                inertia_weight=inertia_weight,
            )

        # elif stochastic is True and not order_test and not two_d:
        elif stochastic is True:
            return StochasticMovement(
                bias=bias,
                f_inertia_force=f_inertial_force,
                b_inertia_force=b_inertial_force,
                inertia_weight=inertia_weight,
            )
        else:
            return DeterministicMovement(step_size=step_size)


class TrophallaxisCreator:
    """Class to select correct forager trophallaxis class based on model paramters

    Attributes
    ---------
    stochastic : bool
        bool to choose for deterministic or stochastic trophallaxis

    Methods
    ------
    __new__(args)
        returns the correct trophallaxis class based on value of stochastic
    """

    def __init__(self, stochastic=True):
        self.stochastic = stochastic

    def __new__(cls, stochastic):
        if stochastic:
            return StochasticTrophallaxis()
        else:
            return DeterministicTrophallaxis()


class Trophallaxis(ABC):
    """Abstract trophallaxis class with common mehtods and necessary methods that differ between child classes

    Methods
    ------
    prop_empty_space_given(agent)
        returns the proportion of the recipient empty crop space the donor gives
    can_agent_give_food(anget)
        returns whether donor has food to give
    is_neighbour_not_full(agent)
        returns whether recipient has space in their crop
    give_food_amount(agent, offered_food)
        returns amount of food actually transfered from donor to recipient
    offered_food(other_agent, variable)
        returns the amount of food offered by the donor
    trophallaxis(agent)
        carries out process of trophallaxis between donor and recipient
    """

    @staticmethod
    def prop_empty_space_given(agent):
        pass

    @staticmethod
    def can_agent_give_food(agent):
        """Returns if donor has food to give in a trophallaxis instance.

        If it is a diffuse test, method checks if ant has more food than the average. In normal cases, method checks if donors crop state is greater than zero.


        Parameters
        ---------
        agent : Ant
            Donor
        """
        if agent.model.diff_test:
            # This was done to test diffusion in the model, where nestmates could only give food above a certain crop to see if diffusion allowed all ants reached said crop
            if isinstance(agent, Nestmate):
                if agent.crop_state > (
                    agent.model.full_nest_ants / agent.model.number_of_ants["Nestmate"]
                ):
                    return True

        # if agent is a forager only care whether it has any food
        else:
            if agent.crop_state > 0:
                return True

    @staticmethod
    def is_neighbour_not_full(agent):
        """Returns whether recipient has space in their crop.

        Checks if crop state is less than 1.

        Parameters
        ---------
        agent : Ant
            Recipient
        """
        if agent.crop_state < 1:
            return True

    @staticmethod
    def give_food_amount(agent, offered_food):
        """Returns amount of food for donor to give to recipient.

        If diffuse test, return minimum the amount of food offered or the difference between nestmate crop and average food in nest. Otherwise, return minimum between donor crop state, offered food and crop space in recipient.

        Parameters
        ---------
        agent : Ant
            Donor
        offered_food : float
            food offered by donor
        """
        if agent.model.diff_test:
            if isinstance(agent, Nestmate):
                return min(
                    (
                        agent.crop_state
                        - (
                            agent.model.full_nest_ants
                            / agent.model.number_of_ants["Nestmate"]
                        )
                    ),
                    offered_food,
                )
        else:
            return min(
                agent.crop_state,
                offered_food,
                (
                    agent.interacting_neighbour.capacity
                    - agent.interacting_neighbour.crop_state
                ),
            )

    @staticmethod
    def offered_food(other_agent, variable):
        """Amount of food offered to the recipient.

        Amount of food equal to the empty space in the recipient ant, multiplied by the proportion given by donor

        Parameters
        ---------
        other_agent : Ant
            Recipient
        variable : float
            proportion of recipeint empty space offered
        """
        return (other_agent.capacity - other_agent.crop_state) * variable

    def trophallaxis(self, agent):
        """Trophallaxis interaction

        Checks if there is a donor recipient pair and then decides how much food the donor gives to the recipient and takes/gives this food to each ant respectively.

        Parameters
        ---------
        agent : Ant
            Donor
        """
        gave_food = False
        proportion_given = self.prop_empty_space_given(agent)
        if agent.interacting_neighbour:
            if (
                isinstance(agent, Nestmate)
                or not isinstance(agent.movement_method, DeterministicMovement)
                or agent.movement_method._new_cell_flag
            ):
                if self.can_agent_give_food(agent):
                    if self.is_neighbour_not_full(agent.interacting_neighbour):
                        offered_food = self.offered_food(
                            agent.interacting_neighbour, proportion_given
                        )
                        agent.food_given = self.give_food_amount(agent, offered_food)
                        agent.interacting_neighbour.crop_state += agent.food_given
                        agent.crop_state -= agent.food_given
                        gave_food = True

        agent.interaction = gave_food


class StochasticTrophallaxis(Trophallaxis):
    """Stochastic version of trophallaxis.

    Proportion of recipient empty space given is sampled from an expoential distribution with mean at agent.proportion_to_give.

    Attributes
    ---------
    name : str
        class name

    Methods
    ------
    prop_empty_space_given(agent)
        Proportion of recipient empty space diven by donor
    """

    name = "StochasticTrophallaxis"

    # Proportion of empty space given is sampled from an exponential distribution with mean defined by the user
    @staticmethod
    def prop_empty_space_given(agent):
        """Proportion of recipient empty space diven by donor. Sampled from an exponential distribution with mean at agent.proportion_to_give

        Parameters
        ---------
        agent : Ant
            Donor
        """
        if (
            agent.proportion_to_give >= 1
        ):  # maybe can make this a string, eg 'all' or 'fill'
            sample = 1
        else:
            sample = np.random.exponential(agent.proportion_to_give)
        return sample

    # Not needed
    def trophallaxis(self, agent):
        super().trophallaxis(agent)


class DeterministicTrophallaxis(Trophallaxis):
    """Determinisitc version of trophallaxis.

    Proportion of recipient empty space given is a constant given by agent.proportion_to_give.

    Attributes
    ---------
    name : str
        class name

    Methods
    ------
    prop_empty_space_given(agent)
        Proportion of recipient empty space diven by donor
    """

    name = "DeterministicTrophallaxis"

    @staticmethod
    def prop_empty_space_given(agent):
        """Proportion of recipient empty space diven by donor. Given by agent.proportion_to_give

        Parameters
        ---------
        agent : Ant
            Donor
        """
        sample = agent.proportion_to_give
        return sample

    # Not needed
    def trophallaxis(self, agent):
        super().trophallaxis(agent)


class Movement(ABC):
    """Abstract forager movement class"""

    @abstractmethod
    def move(self, agent, model):
        pass


class OneDMovement(Movement):
    """Abstract forager movement class for 1D nests

    Methods
    ------
    enter_nest(agent, mode)
        check if forager is at the nest exit, if so it takes a step in
    move_from_edge_of_nest(agent, mode)
        check if forager is at end of nest, if so take a step back
    """

    # If the forager has reached the entrance it always moves into the nest
    @staticmethod
    def enter_nest(agent, model):
        """check if forager is at the nest exit, if so it takes a step in

        Parameters
        ---------
        agent : Forager
            current forager
        model : Model
            model to get nest size
        """
        if agent.pos in model.entrance or agent.position == model.entrance:
            agent.pos[0] = 1
            agent.position[0] = 1
            return True

    # If the forager reaches the end of the nest, it moves one step backwards, 1 step backwards may not be the best option as forager may still get stuck in last two cells when the colony is very full
    @staticmethod
    def move_from_edge_of_nest(agent, model):
        """Check if forager is at end of nest, if so take a step back

        Parameters
        ---------
        agent : Forager
            current forager
        model : Model
            model to get nest size
        """
        if int(agent.pos[0]) >= model.length or int(agent.position[0]) >= model.length:
            agent.pos[0] -= 1
            agent.position[0] -= 1
            return True

    @abstractmethod
    def move(self, agent, model):
        pass


# Deterministic movement in models for which the forager only gives food the first time it enters a new cell
class DeterministicMovement(OneDMovement):
    """Deterministic 1D movement class

    Attributes
    ---------
    name : str
        name of class
    step_size : List[float]
        forager step size above and below the crop threshold
    new_cell_flag : bool
        flag to check if forager has moved to new cell

    Methods
    ------
    get_grid_position(continuous_position, step, operator)
        get new continuous and discrete forager position after movement
    move(self, agent, model)
        movement of forager which depends on whether the forager crop is above or below the threshold
    """

    name = "DeterministicMovement"

    # New cell flag is true on first step in which forager enters a new cell
    def __init__(self, step_size, new_cell_flag=False):
        self._new_cell_flag = new_cell_flag
        self.step_size = step_size

    @staticmethod
    def get_grid_position(continuous_position, step, operator):
        """Get new continuous and discrete forager position after movement.

        Parameters
        ---------
        continuous_position: float
            the continuous mapping of the forager position
        step : float
            movement direction
        operator : callable
            addition of substraction
        """
        new_position = operator(continuous_position, step)
        new_cell = int(operator(continuous_position, step))
        return new_position, new_cell

    # If the foragers previous cell is different to its current cell, new_cell_flag is true. Forager only interacts when new_cell_flag is true, defined in trophallaxis method
    def move(self, agent, model):
        """Carries out forager movement based on forager crop and step sizes

        Checks foragers crop state, move according to the correct step size and then updates forager continuous position. Also checks if forager is at end/start of nest

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """

        if self.enter_nest(agent, model):
            self._new_cell_flag = True
            return

        if agent.crop_state > agent.threshold:
            previous_positions = copy.copy(agent.position)
            # agent.position = [x + y for x, y in zip(agent.position, agent.step_sizes[0])]
            agent.position[0] += self.step_size[0]
            self._new_cell_flag = [np.floor(x) for x in previous_positions] != [
                np.floor(y) for y in agent.position
            ]
            if agent.position[0] > model.length:
                agent.position[0] = model.length

        elif agent.crop_state < agent.threshold:
            previous_positions = copy.copy(agent.position)
            # agent.position = [x + y for x, y in zip(agent.position, agent.step_sizes[1])]
            agent.position[0] += self.step_size[1]
            if agent.position[0] < 0:
                agent.position[0] = 0
            self._new_cell_flag = [np.floor(x) for x in previous_positions] != [
                np.floor(y) for y in agent.position
            ]

        if self.move_from_edge_of_nest(agent, model):
            # agent.threshold = agent.crop_state
            self._new_cell_flag = True


class StochasticMovement(OneDMovement):
    """Stochastic 1D movement class

    Attributes
    ---------
    name : str
        name of class
    bias : List[List[float]]
        probability of forager going forward, staying or backward for above and below crop threshold
    f_inertia_force : float
        inertia force at start of nest
    b_inertia_force : float
        inertia force at end of nest
    inertia_weight : float
        stength of interia

    Methods
    ------
    decide_movement(self, bias)
        return movement direction based on biases
    move(self, agent, model)
        movement of forager which depends on whether the forager crop is above or below the threshold
    """

    name = "StochasticMovement"

    persistence = 0

    def __init__(self, bias, f_inertia_force=0, b_inertia_force=0, inertia_weight=0):
        self.movement_bias = bias
        self.weight = inertia_weight
        self.f_force = f_inertia_force
        self.b_force = b_inertia_force

    def decide_movement(self, bias):
        """Method that decides the foragers next move based on the biases provided. Also returns the persistence force when the forager moves with inertia.

        Parameters
        ---------
        bias : List[List[float]]
            the probabilities for the forager to go forward, stay or backward whe the forager is above and below the crop threshold
        """
        coin = random.random()
        coin += self.persistence * self.weight
        if coin < bias[0]:
            return -1, -coin
        elif bias[0] < coin < bias[1]:
            return 0, 0
        else:
            return 1, coin

    def move(self, agent, model):
        """Movement of forager which depends on whether the forager crop is above or below the threshold

        If forager is at nest exit, it takes a step in and the method ends. Otherwise the forager moves based on her crop. If she steps out the end of the nest, her posiiton is returned to the last position in the nest.

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """

        # Foragers first step into the nest on each visit its persistence is set to the user defined forward force
        if self.enter_nest(agent, model):
            self.persistence = self.f_force
            return

        else:
            if agent.crop_state > agent.threshold:
                direc, self.persistence = self.decide_movement(self.movement_bias[0])
                # self.persistence = direc * self.persistence
                agent.position[0] += direc

            else:
                direc, self.persistence = self.decide_movement(self.movement_bias[1])
                # self.persistence = direc * self.persistence
                agent.position[0] += direc

        # If the forager reaches the edge of the nest, its persistence is set to the user defined backward force
        if self.move_from_edge_of_nest(agent, model):
            # agent.threshold = agent.crop_state
            self.persistence = self.b_force


class SpacelessMovement(OneDMovement):
    """Movement class where the forager is always one step away from the exit.

    Forager decides to exit based on a function of it's crop.

    Methods
    ------
    exit_function(x)
        returns probability for forager to exit given her crop
    forager_exit(self, agent)
        checks whether forager is exiting or not, if so, the foragers position is changed
    move(self, agent, model)
        moves agent 1 step at a time, if it reaches the end of the nest it moves back 4 steps
    """

    @staticmethod
    def exit_function(x):
        """Returns probability for forager to exit given her crop.

        Parameters
        ---------
        x : float
            forager crop
        """
        if x == 0:
            return 1
        else:
            return -np.log(x) / 4

    def forager_exit(self, agent):
        """Checks whether forager is exiting or not, if so, the foragers position is changed

        Parameters
        ---------
        agent : Forager
            moving forager
        """
        coin = random.random()
        prob_at_crop = self.exit_function(agent.crop_state)
        if coin <= prob_at_crop:
            agent.position = [0, 0]
            return True

    def move(self, agent, model):
        """Moves agent 1 step at a time, if it reaches the end of the nest it moves back 4 steps

        If forager is at the exit, method returns

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """

        if self.forager_exit(agent):
            return

        else:
            agent.position[0] += 1
            # If the forager reaches the edge of the nest, it moves 4 cells backwards
            if agent.pos[0] >= model.length or agent.position[0] >= model.length:
                agent.position[0] -= 4


# Extreme movement class, redundent
# class ExtremeMovement(OneDMovement):

#     def __init__(self, bias):
#         self.bias = bias

#     @staticmethod
#     def decide_movement(bias):
#         coin = random.random()
#         if coin < bias[0]:
#             return -1
#         else:
#             return 1

#     def move(self, agent, model):

#         if self.enter_nest(agent, model):
#             return

#         else:
#             if agent.crop_state > agent.threshold:
#                 agent.position[0] += self.decide_movement(self.bias[0])
#             else:
#                 agent.position[0] += self.decide_movement(self.bias[1])

#         if self.move_from_edge_of_nest(agent, model):
#             return


# Forager moves according to average veloicty of the biases defined by the user.
class AverageVelocity(OneDMovement):
    """Average velocity forager movement class.

    Similar to deterministic movement class, however, in this case, forager interacts at every step regardless if on same cell or not.

    Attributes
    ---------
    step_size : List[float]
        step size of forager above and below threshold

    Methods
    ------
    move(self, agent, model)
        moves forager according to step size and forager crop state
    """

    def __init__(self, step_size):
        self.step_size = step_size

    def move(self, agent, model):
        """Moves forager according to step size and forager crop state.

        If forager is at entrance, return. If at end of nest, step back 1 cell.

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """

        if self.enter_nest(agent, model):
            return

        if agent.crop_state > agent.threshold:
            agent.position[0] += self.step_size[0]
        else:
            agent.position[0] += self.step_size[1]
            if agent.position[0] < 0:
                agent.position[0] = 0

        if self.move_from_edge_of_nest(agent, model):
            return


class TwoDMovement(Movement):
    """Two dimensional forager movement abstract class

    Methods
    ------
    enter_nest(agent, model)
        If the forager is at the nest entrance, the forager takes a step in to the nest (on x-axis)
    add_pos(lst1, lst2)
        method to element-wise add two lists
    move_direc(self, x)
        returns excluded Moore's neighbourhood to define possible 2D movement directions (const)
    move_from_edge_of_nest(self, agent, model)
        ensures forager does not go out of bounds
    legal_move(position, model)
        check whether a coordinate is inside the nest
    """

    @staticmethod
    def enter_nest(agent, model):
        """If the forager is at the nest entrance, the forager takes a step in to the nest (on x-axis).

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """
        if agent.pos in model.entrance or agent.position == model.entrance:
            agent.position[0] = 1
            return True

    @staticmethod
    def add_pos(lst1, lst2):
        """Method to element-wise add two lists.

        Parameters
        ---------
        lst1 : List
        lst2 : List
        """
        return list(map(add, lst1, lst2))

    @staticmethod
    def move_direc(direction):
        """Returns excluded Moore's neighbourhood to define possible 2D movement directions (const).

        Parameters
        ---------
        direction : int
            move direction {1: inward, 0: stay, -1: outward}
        """
        moves = [[x, y] for x in range(-1, 2) for y in range(-1, 2)]
        if direction == 1:
            return [a for a in moves if sum(a) > 0]
        elif direction == -1:
            return [a for a in moves if sum(a) < 0]
        else:
            return [a for a in moves if sum(a) == 0]

    def move_from_edge_of_nest(self, agent, model):
        """Ensures forager does not go out of bounds.

        Checks if forager is inbounds, if not, we check random positions until we find one in bounds.

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """

        possible_position = agent.position.copy()

        while not self.legal_move(possible_position, model):
            coord = random.choice([0, 1])
            possible_position[coord] = agent.position[coord] + random.choice([-1, 1])

        agent.pos = possible_position
        agent.position = possible_position
        return True

    @staticmethod
    def legal_move(position, model):
        """Check whether a coordinate is inside the nest.

        Parameters
        ---------
        position : List[int]
            forager position
        model : Model
        """

        # TODO: more efficient if I can use the edges, espeically for big nest, something like
        # Can check if its out of bounds by checking coordinate vs higher/depth of nest
        # if (
        #     position[0] < 0
        #     or position[0] >= nest_depth
        #     or position[1] < 0
        #     or position[1] >= nest_height
        # ):
        # position[0] = min(nest_depth - 0.01, max(0, position[0]))
        # position[1] = min(nest_height - 0.01, max(0, position[1]))

        # Or can make a variable that saves arena, not calculate it every time

        arena = [
            [x, y] for x in range(1, model.length) for y in range(model.nest_height)
        ]
        arena = arena + [model.entrance]

        if position in arena:
            return True
        else:
            return False

    @abstractmethod
    def move(self, agent, model):
        pass


class TDStochasticMovement(TwoDMovement):
    """Stochastic 2D forager movement class

    Attributes
    ---------
    name : str

    Methods
    ------
    decide_movement(self, bias)
        Method that decides the foragers next move based on the biases provided. Also returns the persistence force when the forager moves with inertia.
    move(self, agent, model)
        movement of forager which depends on whether the forager crop is above or below the threshold
    """

    name = "TDStochasticMovement"

    persistence = 0

    def __init__(self, bias, f_inertia_force=0, b_inertia_force=0, inertia_weight=0):
        self.movement_bias = bias
        self.weight = inertia_weight
        self.f_force = f_inertia_force
        self.b_force = b_inertia_force

    def decide_movement(self, bias):
        """Method that decides the foragers next move based on the biases provided. Also returns the persistence force when the forager moves with inertia.

        Parameters
        ---------
        bias : List[List[float]]
            the probabilities for the forager to go forward, stay or backward whe the forager is above and below the crop threshold
        """
        coin = random.random()
        coin += self.persistence * self.weight
        if coin < bias[0]:
            return -1, -coin
        elif bias[0] < coin < bias[1]:
            return 0, 0
        else:
            return 1, coin

    def move(self, agent, model):
        """Movement of forager which depends on whether the forager crop is above or below the threshold

        If forager is at nest exit, it takes a step in and the method ends. Otherwise the forager moves based on her crop. If she steps out the end of the nest, her posiiton is returned to the last position in the nest.

        Parameters
        ---------
        agent : Forager
            moving forager
        model : Model
        """
        if self.enter_nest(agent, model):
            self.persistence = self.f_force
            return

        # Forager chooses a random move out of the moves which correspond to her decision direction (forward, stay, backwards)
        else:
            if agent.crop_state > agent.threshold:
                direc, self.persistence = self.decide_movement(self.movement_bias[0])
                # self.persistence = direc * self.persistence
                agent.position = self.add_pos(
                    agent.position, random.choice(self.move_direc(direc))
                )

            else:
                direc, self.persistence = self.decide_movement(self.movement_bias[1])
                # self.persistence = direc * self.persistence
                agent.position = self.add_pos(
                    agent.position, random.choice(self.move_direc(direc))
                )

        if self.move_from_edge_of_nest(agent, model):
            # agent.threshold = agent.crop_state
            self.persistence = self.b_force
