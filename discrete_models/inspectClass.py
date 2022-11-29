import inspect

# Class which allows for return of all attributes of class
class Inspectable:

    @classmethod
    def inspect_init_arguments(cls):
        return inspect.getfullargspec(cls.__init__)[0]

