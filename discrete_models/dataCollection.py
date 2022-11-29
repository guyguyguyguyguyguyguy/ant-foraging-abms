from antClasses import Forager, Nestmate
import numpy as np


# Override of mesa datacollector to allow for inert nestmates to still be recorded in data collection

from functools import partial
import itertools
from operator import attrgetter
import pandas as pd


class modDataCollector:
    """ Class for collecting data generated by a Mesa model.

    A DataCollector is instantiated with dictionaries of names of model- and
    agent-level variables to collect, associated with attribute names or
    functions which actually collect them. When the collect(...) method is
    called, it collects these attributes and executes these functions one by
    one and stores the results.

    """

    model = None

    def __init__(self, model_reporters=None, agent_reporters=None, tables=None):
        """ Instantiate a DataCollector with lists of model and agent reporters.

        Both model_reporters and agent_reporters accept a dictionary mapping a
        variable name to either an attribute name, or a method.
        For example, if there was only one model-level reporter for number of
        agents, it might look like:
            {"agent_count": lambda m: m.schedule.get_agent_count() }
        If there was only one agent-level reporter (e.g. the agent's energy),
        it might look like this:
            {"energy": "energy"}
        or like this:
            {"energy": lambda a: a.energy}

        The tables arg accepts a dictionary mapping names of tables to lists of
        columns. For example, if we want to allow agents to write their age
        when they are destroyed (to keep track of lifespans), it might look
        like:
            {"Lifespan": ["unique_id", "age"]}

        Args:
            model_reporters: Dictionary of reporter names and attributes/funcs
            agent_reporters: Dictionary of reporter names and attributes/funcs.
            tables: Dictionary of table names to lists of column names.

        Notes:
            If you want to pickle your model you must not use lambda functions.
            If your model includes a large number of agents, you should *only*
            use attribute names for the agent reporter, it will be much faster.
        """
        self.model_reporters = {}
        self.agent_reporters = {}

        self.model_vars = {}
        self._agent_records = {}
        self.tables = {}

        if model_reporters is not None:
            for name, reporter in model_reporters.items():
                self._new_model_reporter(name, reporter)

        if agent_reporters is not None:
            for name, reporter in agent_reporters.items():
                self._new_agent_reporter(name, reporter)

        if tables is not None:
            for name, columns in tables.items():
                self._new_table(name, columns)

    def _new_model_reporter(self, name, reporter):
        """ Add a new model-level reporter to collect.

        Args:
            name: Name of the model-level variable to collect.
            reporter: Attribute string, or function object that returns the
                      variable when given a model instance.
        """
        if type(reporter) is str:
            reporter = partial(self._getattr, reporter)
        self.model_reporters[name] = reporter
        self.model_vars[name] = []

    def _new_agent_reporter(self, name, reporter):
        """ Add a new agent-level reporter to collect.

        Args:
            name: Name of the agent-level variable to collect.
            reporter: Attribute string, or function object that returns the
                      variable when given a model instance.

        """
        if type(reporter) is str:
            attribute_name = reporter
            reporter = partial(self._getattr, reporter)
            reporter.attribute_name = attribute_name
        self.agent_reporters[name] = reporter

    def _new_table(self, table_name, table_columns):
        """ Add a new table that objects can write to.

        Args:
            table_name: Name of the new table.
            table_columns: List of columns to add to the table.

        """
        new_table = {column: [] for column in table_columns}
        self.tables[table_name] = new_table

    def _record_agents(self, model, forag: bool = False):
        """ Record agents data in a mapping of functions and agents. """
        rep_funcs = self.agent_reporters.values()
        if all([hasattr(rep, 'attribute_name') for rep in rep_funcs]):
            prefix = ['model.schedule.steps', 'unique_id']
            attributes = [func.attribute_name for func in rep_funcs]
            get_reports = attrgetter(*prefix + attributes)
        else:
            def get_reports(agent):
                prefix = (agent.model.schedule.steps, agent.unique_id)
                reports = tuple(rep(agent) for rep in rep_funcs)
                return prefix + reports
        if forag:
            agent_records = map(get_reports, model.foragers)
        else:
            agent_records = map(get_reports, model.agents)
        return agent_records

    def collect(self, model, forag: bool = False):
        """ Collect all the data for the given model object. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            agent_records = self._record_agents(model, forag)
            self._agent_records[model.schedule.steps] = list(agent_records)

    def add_table_row(self, table_name, row, ignore_missing=False):
        """ Add a row dictionary to a specific table.

        Args:
            table_name: Name of the table to append a row to.
            row: A dictionary of the form {column_name: value...}
            ignore_missing: If True, fill any missing columns with Nones;
                            if False, throw an error if any columns are missing

        """
        if table_name not in self.tables:
            raise Exception("Table does not exist.")

        for column in self.tables[table_name]:
            if column in row:
                self.tables[table_name][column].append(row[column])
            elif ignore_missing:
                self.tables[table_name][column].append(None)
            else:
                raise Exception("Could not insert row with missing column")

    @staticmethod
    def _getattr(name, object):
        """ Turn around arguments of getattr to make it partially callable."""
        return getattr(object, name, None)

    def get_model_vars_dataframe(self):
        """ Create a pandas DataFrame from the model variables.

        The DataFrame has one column for each model variable, and the index is
        (implicitly) the model tick.

        """
        return pd.DataFrame(self.model_vars)

    def get_agent_vars_dataframe(self):
        """ Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        """
        all_records = itertools.chain.from_iterable(
            self._agent_records.values())
        rep_names = [rep_name for rep_name in self.agent_reporters]

        df = pd.DataFrame.from_records(
            data=all_records,
            columns=["Step", "AgentID"] + rep_names,
        )
        df = df.set_index(["Step", "AgentID"])
        return df

    def get_table_dataframe(self, table_name):
        """ Create a pandas DataFrame from a particular table.

        Args:
            table_name: The name of the table to convert.

        """
        if table_name not in self.tables:
            raise Exception("No such table.")
        return pd.DataFrame(self.tables[table_name])





# Methods in this file are used for data collection in the model

# Colony state, taken to be the average crop of all non-forager ants in the model
def get_colony_state(model):
    number_of_fors = model.nestmate_number
    # total_crop = sum([x.crop_state for x in model.agents if not isinstance(x, Forager)])
    total_crop = np.sum([x.crop_state for x in model.nestmates])
    colony_crop = total_crop/number_of_fors
    return colony_crop


def get_empty_colony_state(model):
    return 1 - get_colony_state(model)


def model_step(model):
    return model.steps


def get_foragers_at_exit(model):
    foragers_at_exit = [x for x in model.agents if isinstance(x, Forager) and x.position ==[0, 0]and x.lag_counter >= 1]

    return foragers_at_exit


def get_exiting_crop(model):
    return [j.exiting_crop for j in get_foragers_at_exit(model)]


def get_steps_to_exit(model):
    return [j.trip_length for j in get_foragers_at_exit(model)]


def get_interacting_forager(forags):
    inter_for = [x for x in forags if x.interaction]
    if inter_for:
        return inter_for
    return None


def get_interacting_forager_crop(forag):
    return [x.crop for x in get_interacting_forager(forag)]


def get_recipients(forag):
    # print(get_interacting_forager(model))
    interacting_forager_position = forag.position
    # recipient = [y for y in forag.model.agents if isinstance(y, Nestmate) and
    #               y.position ==
    #                   interacting_forager_position]
    recipient = forag.model.nestmates_pos_dict[(int(forag.pos[0]),
                                              int(forag.pos[1]))]

    return recipient


def get_interaction_volume(forag):
    volume_given_by_interacting_forager = get_interacting_forager(
        forag).trophallaxis_method.food_given

    return volume_given_by_interacting_forager

