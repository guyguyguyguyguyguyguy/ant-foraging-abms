from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence


def is_number_valid(func):
    def wrapper(self, *args):
        if args[0] == "Forager":
            if int(args[1]) and args[1] >= 0:
                # return func(*args)
                return func(
                    self, key=args[0], number_of_ants=args[1], deployment=args[2]
                )
            else:
                raise Exception(
                    "Number of forager ants entered must be a positive integer"
                )

        elif args[1] == "Nestmate":
            if 0 <= int(args[1]) and args[1] < self.model.nest_depth:
                return func(*args)
            else:
                raise Exception(
                    "Number of forager ants entered must be an integer between 0 and depth of nest"
                )

    return wrapper


def is_string_valid(func):
    def wrapper(self, *args):
        if args[1].lower() == "all":
            return func(self, key=args[0], number_of_ants=args[1], deployment=args[2])
        else:
            raise Exception("Only acceptable string is all")

    return wrapper


def is_float_valid(func):
    def wrapper(self, *args):
        if 0 < args[1] < 1:
            return func(self, key=args[0], number_of_ants=args[1], deployment=args[2])
        else:
            raise Exception("Fraction must be between 0 and 1")

    return wrapper


class Validation(ABC):
    @abstractmethod
    def set_values(self, number_of_ants, deployment):
        pass

    @abstractmethod
    def validate(self, number_of_ants, deployment):
        pass

    @staticmethod
    def int_list_list_lentwo(lst):
        all_elements_are_list = all(isinstance(x, list) and len(x) == 2 for x in lst)
        elements_of_elements_are_int = all(isinstance(b, int) for y in lst for b in y)
        return all_elements_are_list and elements_of_elements_are_int

    @staticmethod
    def allowed_string(string):
        possible_strings = ["random"]
        # noinspection PyBroadException
        try:
            if string.lower() in possible_strings:
                return True
            else:
                return False
        except:
            return False


class OneDValidation(Validation):
    """
    class
    """

    def __init__(self, model):
        self.model = model

    def set_values(self, number_of_ants, deployment) -> Sequence[Dict]:
        if self.validate(number_of_ants, deployment):
            return number_of_ants, deployment

    def validate(self, number_of_ants, deployment):
        try:
            for key, value in number_of_ants.items():
                if deployment is not None:
                    deployment_for_ant = deployment[key]
                else:
                    deployment_for_ant = None
                method_name = type(value).__name__ + "_ant"
                method = getattr(self, method_name, lambda: "Invalid number of ants")
                # method = locals()[method_name]
                return method(key, value, deployment_for_ant)
        except:
            raise Exception(
                "Number of ant entry is not correct, must be an integer, list, or float"
            )

    @is_number_valid
    def int_ant(self, key, number_of_ants, deployment):
        # noinspection PyBroadException
        try:
            if (
                deployment is None
                or self.allowed_string(deployment)
                or self.int_list_list_lentwo(deployment)
            ):
                return True
            else:
                raise Exception
        except:
            return False

    @is_string_valid
    def str_ant(self, key, number_of_ants, deployment):
        Warning("Deployment method will not be taken into consideration")

    @is_float_valid
    def float_ant(self, key, number_of_ants, deployment):
        # noinspection PyBroadException
        try:
            if self.allowed_string(deployment) or deployment is None:
                return True
            else:
                raise Exception("")
        except:
            return False
