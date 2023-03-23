
import logging
import simpy
import simpy.core
import simpy.events
from injector import Module, provider

#====================#
# SIMPY ENVIRONMENTS #
#====================#

class DebugEnvironment(simpy.Environment):
    """Extension of simpy.Environment which prints debug information"""
    
    def __init__(self):
        super().__init__()
    
    def step(self) -> None:
        logging.info(">> step()")
        return super().step()
    
    def schedule(self, event: simpy.Event, priority: simpy.events.EventPriority = simpy.events.NORMAL, delay: simpy.core.SimTime = 0) -> None:
        logging.info(">> schedule(event=%s, priority=%d, delay=%f)", event, priority, delay)
        return super().schedule(event, priority, delay)
    

class SimpyProvider(Module):
    @provider
    def provide_environment(self) -> simpy.Environment:
        return simpy.Environment() if logging.root.level > logging.DEBUG else DebugEnvironment()
