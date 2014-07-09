"""
Parallel Tempering Algorithm
(intended to replace SGD for use with MLPs)

Emily Flynn
7/9/2014

Parallel tempering creates multiple Monte Carlo chains and runs them
at different temperatures, swapping the temperatures of the chains
with a certain probability after they have undergone several Monte
Carlo steps.

An example yaml file is provided: 
     scripts/emily_test/ptemp_test.yaml

To run parallel tempering with this example file:
    python <path_to_scripts>/scripts/train.py <path_to_scripts>/scripts/emily_test/ptemp_test.yaml



--------------------------------------------------------------------
 Note: this code currently uses multiple Train objects to run each
 chain of Monte Carlo. This was done to avoid problems with monitoring
 costs, but should be modified to use Models instead of Train
 objects. The code also needs to be tested to make sure it works with
 a variety of user-specified options.
-------------------------------------------------------------------
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
from pylearn2.training_algorithms.mc import MC
import copy
from pylearn2.train import Train
from pylearn2.termination_criteria import EpochCounter
from datetime import datetime
import os

log = logging.getLogger(__name__)


class PTemp(TrainingAlgorithm):
    """
    Borrows some params from SGD - should consider which we actually
    need and which are extra, and test different parameters to make
    sure that they work.

    """
    def __init__(self, learning_rate, cost=None, batch_size=None,
                 monitoring_batch_size=None, monitoring_batches=None,
                 monitoring_dataset=None, monitor_iteration_mode='sequential',
                 termination_criterion=None, update_callbacks=None,
                 learning_rule=None, init_momentum=None,
                 set_batch_size=False,
                 train_iteration_mode=None, batches_per_iter=None,
                 theano_function_mode=None, monitoring_costs=None,
                 seed=[2012, 10, 5], num_steps=5, 
                 temp_list=[0.2,0.3,0.4], stdev=0.09, k=1,
                 output_file='ptemp_log.txt'):

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

        self.k=k
        self.stdev=stdev
        self.temp_list = temp_list
        self.num_steps = num_steps

        output_path = "%s/%s" %(os.getcwd(), output_file)
        print "output written to %s" %(output_path)
        self.output_log = open(output_path, 'w')

        # write header
        self.output_log.write("epochs\treplicate\ttemp\tenergy\tstdev\tk\n")

        self.replicate_algs = []
        self.replicates = []

        replicate_termination_criterion = EpochCounter(max_epochs = self.num_steps)

        for T in self.temp_list:
            replicate_alg = MC(learning_rate, 
                               cost=cost, batch_size=batch_size,
                               monitoring_batch_size=monitoring_batch_size, 
                               monitoring_batches=monitoring_batches,
                               monitoring_dataset=monitoring_dataset, 
                               monitor_iteration_mode=monitor_iteration_mode,
                               termination_criterion=replicate_termination_criterion, 
                               update_callbacks=update_callbacks,
                               learning_rule=learning_rule,
                               set_batch_size=set_batch_size,
                               train_iteration_mode=train_iteration_mode, 
                               batches_per_iter=batches_per_iter,
                               theano_function_mode=theano_function_mode, 
                               monitoring_costs=monitoring_costs,
                               temp=T, k=self.k, stdev=self.stdev)

            self.replicate_algs.append(replicate_alg)

        return


    def setup(self, model, dataset):
        """
        Sets up parallel tempering by creating each of the train
        objects and running set up on their models and algorithms.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        """

        self.model = model
        self.dataset = dataset

        # Create all of the replicate train objects, each has own
        # model, monitor and algorithm
        for replicate_alg in self.replicate_algs:
            replicate_model = copy.deepcopy(model)

            replicate_monitor = Monitor(replicate_model)
            replicate_model.monitor = replicate_monitor

            replicate_alg.setup(replicate_model, dataset)
            replicate = Train(dataset, 
                              replicate_model,
                              algorithm=replicate_alg)
         
            self.replicates.append(replicate)


        # set up own monitor after
        self._setup_monitor()
        return


    def train(self, dataset):
        """
        Runs training on the dataset using the train object replicates
        and Monte Carlo algorithm.

        """

        # Run x MC steps for each replicate i at temp Ti
        for i,replicate in enumerate(self.replicates):
            print "Started training replicate %d" %(i)
            self.alt_train_loop(replicate)
            print "Done training replicate %d" %(i)
        
        # Make sure that energies have been recorded before doing
        # replicate exchange - should check all!
        if self.replicates[0].algorithm.energy == None:
            return

        # Replicate exchange for all pairs
        for i in range(0, len(self.replicates)):
            for j in range(i+1, len(self.replicates)):
                t1 = self.replicates[i]
                t2 = self.replicates[j] 
                if self._pSwap(t1, t2):
                   self._swapTemp(i, j)

        epochs = self.monitor.get_epochs_seen()
        for i,replicate in enumerate(self.replicates):
            print "Replicate %i with temp %.4f and energy %.4f" %(i, replicate.algorithm.temp, replicate.algorithm.energy) 

            output_string = "%i\t%i\t%.4f\t%.5f\t%.3f\t%.3f\n" %(epochs, i, replicate.algorithm.temp, replicate.algorithm.energy, self.stdev, self.k)
            #print "output string: %s" %(output_string) 
            self.output_log.write(output_string)


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
                self.output_log.close()
                return False



    def _pSwap(self, rep1, rep2):
        """
        Calculate the probability of swapping two replicates.
        """
        
        
        t1 = rep1.algorithm
        t2 = rep2.algorithm

        p = min(1, (cmath.e)**((t1.energy-t2.energy)*((1.0/(self.k*t1.temp)) - (1.0/(self.k*t2.temp)))))

        rval = random.uniform(0,1)
        
        if p < rval:
            return True
        
        return False



    def _swapTemp(self, i, j):
        """
        Function to swap the temps of the ith and jth replicate
        objects. 
        """
        print "Swapping replicates %i and %i\n" %(i,j)

        holder = self.replicates[i].algorithm.temp
        self.replicates[i].algorithm.temp = self.replicates[j].algorithm.temp
        self.replicates[j].algorithm.temp = holder
            
        return

    
    def _setup_monitor(self):
        """ 
        Code to set up monitor. Same as that used in sgd.py
        """

        if self.cost is None:
            self.cost = self.model.get_default_cost()

        inf_params = [param for param in self.model.get_params()
                      if np.any(np.isinf(param.get_value()))]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([np.any(np.isnan(param.get_value()))
                for param in self.model.get_params()]):
            nan_params = [param for param in self.model.get_params()
                          if np.any(np.isnan(param.get_value()))]
            raise ValueError("These params are NaN: "+str(nan_params))
      
        self.monitor = Monitor.get_monitor(self.model)
        self.monitor._sanity_check()

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
        fixed_var_descr = self.cost.get_fixed_var_descr(self.model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch


        cost_value = self.cost.expr(self.model, nested_args,
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
        
        return


    def alt_train_loop(self, replicate):

        """ 
        Does same thing as main_loop in train.py, except doesn't call
        setup(). This is necessary because we are running each train
        object multiple times, and we cannot set it up
        repeatedly. Note that this train_loop does not monitor as the
        process as closely as the original - we do not keep track of
        timing, etc, to simplify the process. 
        """

        time_budget = None
        t0 = datetime.now()
        replicate.run_callbacks_and_monitoring()
        while True:
            if replicate.exceeded_time_budget(t0, time_budget):
                break
            
            with log_timing(log, None, level=logging.DEBUG,
                            callbacks=[replicate.total_seconds.set_value]):
                with log_timing(
                    log, None, final_msg='Time this epoch:',
                    callbacks=[replicate.training_seconds.set_value]):
                    rval = replicate.algorithm.train(dataset=replicate.dataset)
                    if rval is not None:
                        raise ValueError("TrainingAlgorithm.train should not "
                                         "return anything. Use "
                                         "TrainingAlgorithm.continue_learning "
                                         "to control whether learning "
                                         "continues.")
                replicate.model.monitor.report_epoch()
                extension_continue = replicate.run_callbacks_and_monitoring()
                if replicate.save_freq > 0 and \
                        replicate.model.monitor.get_epochs_seen() % replicate.save_freq == 0:
                        replicate.save()
            continue_learning = (
                replicate.algorithm.continue_learning(replicate.model) and
                extension_continue
                )
            assert continue_learning in [True, False, 0, 1]
            if not continue_learning:
                break

        replicate.model.monitor.training_succeeded = True
        return

