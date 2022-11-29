import glob
import random
from inspect import getmembers, isroutine
from operator import add, itemgetter

import matplotlib.pyplot as plt
import numpy as np

# from FoodModel import TwoDModel


# Method returns mutual attributes and their values between model and ant subclasses
def get_properties(model, subclass):
    model_attributes = model.inspect_init_arguments()
    properties = [
        getattr(model, x)
        for x in subclass.inspect_init_arguments()
        if x in model_attributes and x != "self"
    ]
    return properties


def figure_number():
    number = (i for i in range(100))
    for no in number:
        yield plt.figure(no)


def powerla(x, a, b):
    return b * np.power(x, a)


def inverse(x, a, b):
    return b * 1 / (x * a)


def func2(x, a, b, c):
    return a * np.exp(x * c) + b


def func1(x, a, b):
    return (1 / (1 - x * a)) * b


def logistic_fun(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


def linear_fun(x, a, b):
    return a + b * x


def open_csv_files(data_directory=None):
    path = "Data/" + data_directory + "/"
    all_files = glob.glob(path + "/*.csv")
    return all_files


def figure_name():
    names = (x for x in range(100))
    for x in names:
        yield "f" + str(x)


def decision_making(probability):
    return np.random.random() <= probability


# Merge model and agent data
def sort_repeated_data(model_data, agent_data, threshold=None, drop_na=False):

    indexed_model_data = model_data.set_index("step")
    indexed_agent_data = agent_data.set_index("step")

    merged_data = indexed_model_data.merge(indexed_agent_data, how="inner", on="step")
    if drop_na:
        merged_data = merged_data.dropna(thresh=threshold)

    return merged_data


def get_nest_ants(model, class_name):
    nest_ants = [x for x in model.agents if isinstance(x, class_name)]

    return nest_ants


# Nestmate movement in models where nestmates are able to move anywhere in the nest at every forager exit (now depreciated due to space_model method)
def nestmate_movement(model, class_name):
    nest_ants = get_nest_ants(model, class_name)
    selected_numbers = [
        random.uniform(x.crop_state, 1) + np.log(x.position[0] + 1) for x in nest_ants
    ]
    sorted_ants = [
        x
        for (_, x) in sorted(zip(selected_numbers, nest_ants), key=lambda pair: pair[0])
    ]

    for n, ant in enumerate(sorted_ants):
        ant.position[0] = n + 1


# Nestmate movement in models where nestmates are shuffled locally at every forager exit, area of shuffle is user defined

# Can also include 'biased' nestmates movement in which nestmates move towards an the entrance if their crop is below a user defined threshold

# Velocity is the number of cells the forager can shuffle within
def nestmate_movement_limited(model, class_name, threshold, velocity, bias=False):
    def add_pos(lst1, lst2):
        return list(map(add, lst1, lst2))

    nest_ants = get_nest_ants(model, class_name)

    if model.name == "one_d_model":

        if not bias:
            sampling_ants = [
                [
                    x,
                    random.randint(
                        max(x.position[0] - velocity, 1),
                        min(x.position[0] + velocity, model.nest_depth),
                    ),
                ]
                for x in nest_ants
                if x.crop_state < threshold
            ]
        else:
            sampling_ants = [
                [x, random.randint(max(x.position[0] - velocity, 1), x.position[0])]
                for x in nest_ants
                if x.crop_state < threshold
            ]

        # sampling_ants = [[i[0], 1] if i[1] < 1 else [i[0], model.nest_depth] if i[1] >= model.nest_depth else i for i in sampling_ants]
        non_sampling = [
            [x, x.position[0]] for x in nest_ants if x.crop_state >= threshold
        ]

        sampling_ants.extend(non_sampling)

        new_positions = sorted(sampling_ants, key=lambda pair: pair[1])

        for n, ant in enumerate(new_positions):
            ant[0].position[0] = n + 1

    if model.name == "two_d_model":
        # Maybe velocity determines how many times this is done per step and each ant moves up to one cell per loop
        # No bias yet, this is quite fast and can be used for 1D too, just doesn't have bias
        d = {tuple(a.pos): a for a in model.nestmates}
        # maybe random order
        for a in model.nestmates:
            move_vec = np.array(random.sample([-velocity, 0, velocity], 2))
            new_pos = a.pos + move_vec
            if new_pos[0] < model.exit_size:
                new_pos[0] = 1
            if new_pos[1] < 0:
                new_pos[1] = 0
            if new_pos[0] >= model.length:
                new_pos[0] = model.length - 1
            if new_pos[1] >= model.nest_height:
                new_pos[1] = model.nest_height - 1
            if d.get(tuple(new_pos)):
                o = d[tuple(new_pos)]
                d[tuple(a.pos)] = o
                o.pos = list(a.pos)
            else:
                del d[tuple(a.pos)]
            d[tuple(new_pos)] = a
            a.pos = list(new_pos)


# function that moves ant within it's Moore-neighborhood, given its velocity
# need to work on what positions are equal or not in 2d
# def 2d_nestmate_move(ant, new_pos, free_spaces):
#    if new_ant_pos in free_spaces:
#        ant.position = free_spaces
#        free_spaces.remove(new_ant_pos)
#    else:
#        pass


# Movement of nestmates in models in which they are shuffled randomly at every forager exit
def space_model(model, class_name):
    nest_ants = get_nest_ants(model, class_name)
    if model.name == "two_d_model":
        new_positions = [
            [x, y] for x in range(1, model.length) for y in range(model.nest_height)
        ]

        for n, ant in enumerate(nest_ants):
            ant.position = new_positions[n]
    else:
        new_positions = random.sample(range(1, model.nest_depth + 1), len(nest_ants))

        for n, ant in enumerate(nest_ants):
            ant.position[0] = new_positions[n]


# Not used
def order_model(model, class_name):
    nest_ants = get_nest_ants(model, class_name)
    nest_ants = nest_ants.sort(key=lambda x: x.position)

    pass
