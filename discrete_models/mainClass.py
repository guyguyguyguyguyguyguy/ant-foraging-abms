from antMethodsClasses import *
from modelClasses import *
import os
import pandas as pd
import time
from helperFunctions import *
from multiprocessing import Pool
import itertools
from copy import deepcopy


#Todo: find referenced parameter that is being changed during course of
# simulation that is altering foragers starting position


# Main class in which model is instantiated with user defined parameters
class Main:

    def __init__(self, steps, inertia, f_inertial_force, b_inertial_force, bias_above, bias_below, step_sizes, repeat_no=7, save=False, verbose=False,
                 order_test=False, space_test=False, homogenise=False, nestmate_movement = False, nest_depth=45, nest_height= 1,
                 stoch_troph = True, stoch_mov = True, lag_length = 1, nestmate_bias =False, propagate_food_rate=0, two_d = False,
                 extreme_move = False, aver_veloc=False, inert_nestants=True, threshold=0.3, deployment= None, number_of_ants= None,
                 multiprocessing=1):
        self.steps=steps
        self.inertia = inertia
        self.f_inertial_force = f_inertial_force
        self.b_inertial_force = b_inertial_force
        self.repeats = repeat_no
        self.save = save
        self.verbose = verbose
        self.order_test = order_test
        self.space_test = space_test
        self.homogenise = homogenise
        self.nestmate_movement = nestmate_movement
        self.nest_depth = nest_depth
        self.nest_height = nest_height
        self.stoch_troph = stoch_troph
        self.stoch_mov = stoch_mov
        self.lag_length = lag_length
        self.nestmate_bias = nestmate_bias
        self.propagate_food_rate = propagate_food_rate
        self.two_d = two_d
        self.extreme_move = extreme_move
        self.bias_above = bias_above
        self.bias_below = bias_below
        self.aver_veloc = aver_veloc
        self.step_sizes = step_sizes
        self.inert_nestants = inert_nestants
        self.threshold=threshold
        self.deployment = deployment
        if number_of_ants is None:
            self.number_of_ants = {'Forager': 1, 'Nestmate': 'all'}
        else:
            self.number_of_ants = number_of_ants
        # pool = Pool(multiprocessing)
        # TODO: fix parralelisation
        # pool.map(self.run())
        self.run()
        self.bias_dataframe =  self.bias_calculator(self.all_forager_data)

    @staticmethod
    def bin_location(x):
        return np.ceil(x/3)

    def bias_calculator(self, forager_data):

        df = pd.DataFrame()
        df['Crop Chunks'] = forager_data.crop.round(1)
        df['step_direction'] = 'NaN'
        if not self.two_d:
            df['step_direction'].iloc[np.argwhere(np.diff(forager_data['position']) > 0).flatten()] = 'inward'
            df['step_direction'].iloc[np.argwhere(np.diff(forager_data['position']) == 0).flatten()] = 'stay'
            df['step_direction'].iloc[np.argwhere(np.diff(forager_data['position']) < 0).flatten()] = 'outward'

            df['Location bin'] = forager_data.position

        else:
            helpp = forager_data['position'].apply(lambda x: sum(x))
            df['step_direction'].iloc[np.argwhere(np.diff(helpp) > 0).flatten()] = 'inward'
            df['step_direction'].iloc[np.argwhere(np.diff(helpp) == 0).flatten()] = 'stay'
            df['step_direction'].iloc[np.argwhere(np.diff(helpp) < 0).flatten()] = 'outward'

            df['Location bin'] = helpp


        df = df[df['Location bin'] > 0]

        df['Location bin'] = df['Location bin'].apply(self.bin_location)

        c_chunk = df.pivot_table(index = ['Crop Chunks', 'Location bin'], columns = 'step_direction',
                                 aggfunc= 'size', fill_value=0)

        if 'Nan' in c_chunk:
            c_chunk = c_chunk.drop('NaN', axis=1)
        c_chunk = c_chunk.div(c_chunk.sum(axis=1), axis=0)

        return c_chunk


    def run(self):

        trophallaxis_method = None
        movement_method = None

        self.all_step_data_average = pd.DataFrame()
        self.all_interaction_data = pd.DataFrame()
        self.all_visit_data = pd.DataFrame()
        self.all_forager_data = pd.DataFrame()


        # if self.extreme_move:
        #     forager_bias = [self.bias_above, self.bias_below]
        # else:
        #     forager_bias = [[0.3, 0.65, 1], [0.5, 0.75, 1]]

        stoch_mov = self.stoch_mov
        forager_bias = [self.bias_above, self.bias_below]
        step_sizes = self.step_sizes
        inertia = self.inertia
        f_inertial_force = self.f_inertial_force
        b_inertial_force =self.b_inertial_force

        # movement_list = [StochasticMovement(bias=forager_bias), DeterministicMovement()]
        # trophallaxis_list = [StochasticTrophallaxis(),
        #                      DeterministicTrophallaxis()]
        #
        # permutations = [[x,y] for x in movement_list for y in trophallaxis_list]
        # motion_threshold = [x for x in np.arange(0,1,0.1)]
        combination = [MovementCreator(bias=forager_bias, f_inertial_force = f_inertial_force,
                                       b_inertial_force = b_inertial_force, inertia_weight= inertia,
                                       stochastic=stoch_mov, order_test=self.order_test, two_d=self.two_d,
                                       extreme_move= self.extreme_move, aver_veloc=self.aver_veloc, step_size=step_sizes),
                       TrophallaxisCreator(stochastic=self.stoch_troph)]

        tic = time.time()

        for i in range(self.repeats):
            print('\t Repeat number %s' %(i+1) )
            # step_sizes = [[0.05,0],[-0.25,0]]
            forager_biases = forager_bias
            number_of_ants = {'Forager': 1,
                              'Nestmate': 'all'}

            deployment = {'Nestmate': [[1,0], [2,0]],
                          'Forager': None}

            # positions = {Forager: [[0,0]],
            #              Nestmate: [[0,1], [0,2.1]]}

            nest_depth = self.nest_depth
            nest_height =self.nest_height
            exit_size = 1
            #todo: for now this is considered to be interaction rate, but want to change this to be true/false and have
            # an interaction rate defined in the nest ants class, with a value/function defined by the data
            # can do something such as a space dependent, time dependant or crop dependant
            propagate_food = self.propagate_food_rate
            motion_threshold = self.threshold
            ants_homogenise = self.homogenise
            nestmates_can_move = self.nestmate_movement
            lag_legnth = self.lag_length
            diff_test = False
            if diff_test:
                full_nest_ants = 1
            else:
                full_nest_ants = 0

            interaction_rate = 1

            max_steps = self.steps

            # noinspection PyBroadException
            movement_method = combination[0]
            trophallaxis_method = combination[1]

            # print('forager_bias {}'.format(forager_bias))
            # print("movement method {}".format(movement_method))

            # print("nest depth: {}, exit_size: {}, step_sizes = {}, motion_threshold = {}, number_of_ants = {}, trophallaxis = {}, "
            #       "movement: {}, propogatefood: {}, maxsteps: {}, deployment: {}, homogenise: {}, nestmove: {}, foragerlag: {}, "
            #       "diff: {}, fullnestants: {}, forag_int_rate: {}, repeats:{}, verbose:{}, space_test:{}, nestmatebias: {}, "
            #       "inertnestnts:{}".format(nest_depth, exit_size, step_sizes, motion_thresh, number_of_ants, trophallaxis_method, movement_method,
            #                                propagate_food, max_steps, deployment, ants_homogenise, nestmates_can_move, lag_legnth, diff_test,
            #                                full_nest_ants, interaction_rate, i+1, self.verbose, self.space_test, self.nestmate_bias,
            #                                self.inert_nestants))


            if not self.two_d:
                self.model = deepcopy(OneDModel(nest_depth=nest_depth,
                                        exit_size=exit_size, step_sizes=step_sizes, motion_threshold=motion_threshold,
                              number_of_ants=self.number_of_ants, trophallaxis_method=trophallaxis_method,
                              movement_method=movement_method, propagate_food=propagate_food, max_steps=max_steps,
                              deployment=self.deployment, homogenise=ants_homogenise, nestmate_movement=nestmates_can_move,
                              forager_lag=lag_legnth, diff_test=diff_test, full_nest_ants=full_nest_ants,
                              forager_interaction_rate=interaction_rate, repeat= (i+1), verbose = self.verbose,
                               space_test = self.space_test, nestmate_bias =
                                                self.nestmate_bias, inert_nestants=self.inert_nestants))
            else:
                self.model= TwoDModel(nest_depth=nest_depth, nest_height=nest_height, exit_size=exit_size, step_sizes=step_sizes, motion_threshold=motion_threshold,
                              number_of_ants=self.number_of_ants, trophallaxis_method=trophallaxis_method,
                              movement_method=movement_method, propagate_food=propagate_food, max_steps=max_steps,
                              deployment=self.deployment, homogenise=ants_homogenise, nestmate_movement=nestmates_can_move,
                              forager_lag=lag_legnth, diff_test=diff_test, full_nest_ants=full_nest_ants,
                              forager_interaction_rate=interaction_rate, repeat= (i+1), verbose = self.verbose,
                               space_test = self.space_test, nestmate_bias = self.nestmate_bias, inert_nestants=self.inert_nestants)


            self.model.populate()
            self.model.run()


            # Merge agent and model dataframes, append this to each (visit, interaction, forager) dataframe after each run of model
            agent_step_data = self.model.step_data_collector.get_agent_vars_dataframe()
            model_step_data = self.model.step_data_collector.get_model_vars_dataframe()
            repeat_model_data = model_step_data.set_index('step')
            repeat_agent_data = agent_step_data.set_index('step')
            repeat_data = repeat_model_data.merge(repeat_agent_data, how='inner', on='step')

            # Average step dataframe over all runs of model
            if i == 0:
                self.all_step_data_average = repeat_data
            else:
                self.all_step_data_average = self.all_step_data_average.add(repeat_data, axis='index', fill_value=None)

            if not diff_test:
                model_interaction_data = self.model.interaction_data_collector.get_model_vars_dataframe()
                agent_interaction_data = self.model.interaction_data_collector.get_agent_vars_dataframe()
                self.all_interaction_data = self.all_interaction_data.append(sort_repeated_data(model_interaction_data,agent_interaction_data, threshold=5, drop_na= True))


                model_visit_data = self.model.visit_data_collector.get_model_vars_dataframe()
                agent_visit_data = self.model.visit_data_collector.get_agent_vars_dataframe()
                if not model_visit_data.empty:
                    self.all_visit_data = self.all_visit_data.append(sort_repeated_data(model_visit_data, agent_visit_data, threshold=4, drop_na=True))

                model_forager_data = self.model.forager_data_collector.get_model_vars_dataframe()
                agent_forager_data = self.model.forager_data_collector.get_agent_vars_dataframe()
                self.all_forager_data = self.all_forager_data.append(sort_repeated_data(model_forager_data, agent_forager_data, threshold=4, drop_na=True))

        if self.repeats > 1:
            self.all_step_data_average = self.all_step_data_average.div(self.repeats)

        toc = time.time()

        print(toc-tic)

