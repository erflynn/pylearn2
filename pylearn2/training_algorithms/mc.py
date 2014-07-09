"""
Monte Carlo Training Algorithm 
(used instead of Gradient Descent)

Emily Flynn
7/9/2014


An example yaml file is provided: 
     scripts/emily_test/mc_test.yaml

To run Monte Carlo with this example file:
    python <path_to_scripts>/scripts/train.py <path_to_scripts>/scripts/emily_test/mc_test.yaml

"""

import logging
import warnings
import numpy as np

from theano import config
from theano import function
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values

from pylearn2.monitor import Monitor
from pylearn2.space import CompositeSpace, NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils.iteration import is_stochastic, has_uniform_batch_size
from pylearn2.utils import py_integer_types, py_float_types
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils import sharedX
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.timing import log_timing
from pylearn2.utils.rng import make_np_rng

###
from pylearn2.utils import as_floatX
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T 
import cmath
import random 



log = logging.getLogger(__name__)

class MC(TrainingAlgorithm):

    """
    Currently using some of the same parameters as SGD...  Need to
    rewrite and consider what is necessary, also make sure options
    work for these parameters.
    
    """
    def __init__(self, learning_rate, cost=None, batch_size=None,
                 monitoring_batch_size=None, monitoring_batches=None,
                 monitoring_dataset=None, monitor_iteration_mode='sequential',
                 termination_criterion=None, update_callbacks=None,
                 learning_rule=None, init_momentum=None,
                 set_batch_size=False,
                 train_iteration_mode=None, batches_per_iter=None,
                 theano_function_mode=None, monitoring_costs=None,
                 seed=[2012, 10, 5], temp=0.2, k=1, stdev=0.09):

        self.learning_rule = learning_rule
        self.learning_rate = sharedX(learning_rate, 'learning_rate')
        self.cost = cost
        self.batch_size = batch_size
        self.set_batch_size = set_batch_size
        self.batches_per_iter = batches_per_iter
        self._set_monitoring_dataset(monitoring_dataset)
        self.monitoring_batch_size = monitoring_batch_size
        self.monitoring_batches = monitoring_batches
        self.monitor_iteration_mode = monitor_iteration_mode
        if monitoring_dataset is None:
            if monitoring_batch_size is not None:
                raise ValueError("Specified a monitoring batch size " +
                                 "but not a monitoring dataset.")
            if monitoring_batches is not None:
                raise ValueError("Specified an amount of monitoring batches " +
                                 "but not a monitoring dataset.")
        self.termination_criterion = termination_criterion
        self._register_update_callbacks(update_callbacks)
        if train_iteration_mode is None:
            train_iteration_mode = 'shuffled_sequential'
        self.train_iteration_mode = train_iteration_mode
        self.first = True
        self.rng = make_np_rng(seed, which_method=["randn","randint"])
        self.theano_function_mode = theano_function_mode
        self.monitoring_costs = monitoring_costs

        self.obj_channel = "train_objective"
        self.recorded_cost = []
                
        self.k = k
        self.stdev = stdev
        self.temp = temp 
        self.energy = None
        self.is_setup = False

        return

    def setup(self, model, dataset):
        """
        Compiles the theano functions needed for the train method.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        """
        
        if self.is_setup:
            return 

        self.model = model

        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [param for param in model.get_params()
                      if np.any(np.isinf(param.get_value()))]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value()))
                for param in model.get_params()]):
            nan_params = [param for param in model.get_params()
                          if np.any(np.isnan(param.get_value()))]
            raise ValueError("These params are NaN: "+str(nan_params))
      
        self.monitor = Monitor.get_monitor(model)

        self.monitor._sanity_check()

        # test if force batch size and batch size
        if getattr(model, "force_batch_size", False) and \
           any(dataset.get_design_matrix().shape[0] % self.batch_size != 0 for
               dataset in self.monitoring_dataset.values()) and \
           not has_uniform_batch_size(self.monitor_iteration_mode):

            raise ValueError("Dataset size is not a multiple of batch size."
                             "You should set monitor_iteration_mode to "
                             "even_sequential, even_shuffled_sequential or "
                             "even_batchwise_shuffled_sequential")

        data_specs = self.cost.get_data_specs(self.model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name,
                                          batch_size=self.batch_size)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = self.cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch


        cost_value = self.cost.expr(model, nested_args,
                                    ** fixed_var_descr.fixed_vars)

        if cost_value is not None and cost_value.name is None:
            # Concatenate the name of all tensors in theano_args !?
            cost_value.name = 'objective'

        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost
        learning_rate = self.learning_rate
        if self.monitoring_dataset is not None:
            if (self.monitoring_batch_size is None and
                    self.monitoring_batches is None):
                self.monitoring_batch_size = self.batch_size
                self.monitoring_batches = self.batches_per_iter
            
                self.monitor.setup(dataset=self.monitoring_dataset,
                               cost=self.cost,
                               batch_size=self.monitoring_batch_size,
                               num_batches=self.monitoring_batches,
                               extra_costs=self.monitoring_costs,
                               mode=self.monitor_iteration_mode)
            dataset_name = self.monitoring_dataset.keys()[0]
            monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            self.monitor.add_channel(name='learning_rate',
                                     ipt=None,
                                     val=learning_rate,
                                     data_specs=(NullSpace(), ''),
                                     dataset=monitoring_dataset)

            if self.learning_rule:
                self.learning_rule.add_channels_to_monitor(
                        self.monitor,
                        monitoring_dataset)
                
        

        # set up pre-compiled theano functions - should reduce to a
        # single function
        self.srng = RandomStreams(seed=702) ## should change!!!
        

        params = list(model.get_params())
        assert len(params) > 0
        
        self.updates_dict = dict() #
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'mc_params[%d]' % i
            value = param.get_value(borrow=True)
            rv_g = self.srng.normal(param.shape, std=self.stdev)

            ### there should be a simpler way to do this - maybe
            ### set type of x to type of value??
            if len(value.shape) <= 1:
                x = T.dvector('x')
            else:
                x = T.dmatrix('x')
                
            z = rv_g + x
            f = function([x], z)
            self.updates_dict[param.name] = f

        self.params = params
        self.is_setup=True

        return
    

    def train(self, dataset):
        """
        Training function for Monte Carlo.
        """


        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        
        # Random displacement
        old_params = dict()
        for param in self.params:
            param_value = param.get_value(borrow=True)
            old_params[param.name]=param_value

            new_value = as_floatX(self.updates_dict[param.name](as_floatX(param_value)))
            param.set_value(new_value)
            
            
        #  Accept/Reject by Metropolis
        if (not self._metropolisAccept()):
            # if reject by metropolis, set back to old_params  
            for param in self.params:
                param.set_value(old_params[param.name])
            

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        return




    def continue_learning(self, model):
        """
        Returns True if the algorithm should continue running, or False
        if it has reached convergence / started overfitting and should
        stop.

        Parameters
        ----------
        model : a Model instance
        """
        if self.termination_criterion is None:
            return True
        else:
            if self.termination_criterion.continue_learning(self.model):
                return True
            else:
                
                # write out the cost
                #np.savetxt("cost.csv", self.recorded_cost)
                return False


    def _metropolisAccept(self):
        """
        Function to accept/reject the next step based on
        Metropolis-Hastings.
        """

        cost = self.monitor.channels[self.obj_channel].val_record

        if len(cost) == 0:            
            return True

        if len(cost) ==1:
            self.recorded_cost.append(cost[-1]) 
            self.energy = cost[-1]
            return True


        candidate_cost = cost[-1]
        current_cost = self.energy 

        print "Candidate Cost: %.5f" %(candidate_cost)
        print "Current Cost: %.5f" %(current_cost)

        delta_cost = candidate_cost - current_cost

        # fuzzy inequality - check
        if (delta_cost + 0.000001) < 0:
            print "Accepted - better cost"
            self.recorded_cost.append(candidate_cost)
            self.energy = candidate_cost
            return True

        else:
            
            w = cmath.e**(-delta_cost/(self.k*self.temp))

            ## difference between random.random() and random.uniform()??
            rval = random.uniform(0,1) 

            print "w = %.5f, rval= %.3f" %(w,rval)

            if w > rval:
                print "Accepted"
                self.recorded_cost.append(candidate_cost)
                self.energy = candidate_cost
                return True
  
            else:
                print "Rejected"
                self.recorded_cost.append(current_cost)
                self.energy = current_cost
                return False
        


