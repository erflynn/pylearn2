"""
Monte Carlo Training Algorithm
(used instead of Gradient Descent)

Emily Flynn
7/16/2014


An example yaml file is provided:
     scripts/emily_test/mc_test.yaml

To run Monte Carlo with this example file:
    python <path_to_scripts>/scripts/train.py 
      <path_to_scripts>/scripts/emily_test/mc_test.yaml

"""

import logging
import numpy as np

from theano import function
from pylearn2.monitor import Monitor
from pylearn2.space import NullSpace

from pylearn2.training_algorithms.training_algorithm import TrainingAlgorithm
from pylearn2.utils.iteration import has_uniform_batch_size
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX
from pylearn2.utils.data_specs import DataSpecsMapping
from pylearn2.utils.rng import make_np_rng

###
from pylearn2.utils import as_floatX
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T 
import cmath
import random 
from datetime import datetime
import os

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
                 seed=[2012, 10, 5], temp=0.2, k=1, stdev=0.09, 
                 update_entries=None, layer_by_layer=False):

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
        self.num_accepts = 0   # number of accepts out of comparisons (not total!)
        self.num_rejects = 0

        #### parse the "update_entries" YAML
        ## should clean up, set up defaults
        # set up so that we can change, select different defaults
        if (update_entries is None):
            update_entries = dict() 
            update_entries['method'] = "keep"
        elif ('method' not in update_entries):
            update_entries['method'] = "keep"
        elif update_entries['method'] == "by_num":
            if 'num_entries' not in update_entries:
                self.num_entries_to_keep = 10
            else:
                self.num_entries_to_keep = update_entries['num_entries']         
        elif update_entries['method'] == "by_fraction":
            if 'fraction_entries' not in update_entries:
                self.fraction_entries = 0.10
            else:
                self.fraction_entries = update_entries['fraction_entries']     
        #else:
            # do nothing for now
            
        self.update_method = update_entries["method"]
                

        self.layer_by_layer = layer_by_layer
        self.layer_idx = 0


    def setup(self, model, dataset):
        """
        Compiles the theano functions needed for the train method.

        Note: this method should set up a single pre-compiled theano
        function to apply to all parameters, but currently creates an
        "updates_dict" with one update function associated with each
        parameter.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        """
        
        if self.is_setup:
            return 

        self.model = model
        
        # runs function to check settings and set up monitor
        self._settings_and_monitor()

        # seed the random number generator
        # note: should switch to pylearn2 rng seed functions
        self.srng = RandomStreams(seed=datetime.now().microsecond) 

        # run method to set up sample matrices
        self._set_up_sample_matrices()
    

        ###########################################################
        # Set up the update function:
        #
        #  the update function for each entry in the matrix is:
        #    new_value = original_value + random_displace*update_bool
        #
        #  the "random_displace" is a gaussian random displacement
        #  with a mean=0 and stdev specified by the YAML file
        #
        #  the "update_bool" comes from the sample_matrix and is
        #  a 1 if the value is updated and a zero if not. this 
        #  provides a way for only a portion of the weights to be
        #  updated at each step.
        #
        ##########################################################
        
        self.updates_dict = dict() 
        for i, param in enumerate(self.params):
            if param.name is None:
                param.name = 'mc_params[%d]' % i
            value = param.get_value(borrow=True)
            rv_g = self.srng.normal(param.shape, std=self.stdev)

            ### there should be a simpler way to do this - maybe
            ### set type of x,y to type of value??
            if len(value.shape) <= 1:
                x = T.dvector('x')   ### d prefix correct???
                y = T.dvector('y')
            else:
                x = T.dmatrix('x')
                y = T.dmatrix('y')

            z = x + rv_g*y 
            f = function([x, y], z)
            self.updates_dict[param.name] = f


        self.is_setup=True


    def train(self, dataset):
        """
        Training function for Monte Carlo.
        """

        # Make sure none of the parameters have bad values
        self._check_param_values(self.params)
        


        # option to only update one layer each epoch        
        if (self.layer_by_layer):
            
            current_layer = self.model.layers[self.layer_idx]
            
            # train the parameters of the current layer
            self._train_params(current_layer.get_params())
            
            # update the layer index
            self.layer_idx = (self.layer_idx+1)%(len(self.model.layers))
        
    
        else:
            self._train_params(self.params)
            
                

            
        """
        # Random displacement
        old_params = dict()
        for param in self.params:

            param_value = param.get_value(borrow=True)

            # save a copy of the old value
            old_params[param.name]=param_value

            # use displacement of a subset of the entries to get a
            # matrix of new weight values for the parameter
            sample_matrix = self.sample_matrix_dict[param.name]
            new_value = as_floatX(self.updates_dict[param.name]\
                                      (as_floatX(param_value), sample_matrix))
            
            # updates the value of the parameter and the cost 
            param.set_value(new_value) 
            
            
        #  Accept/Reject by Metropolis
        if (not self._metropolis_accept()):
            # if reject by metropolis, set back to the value in the old_params
            for param in self.params:
                param.set_value(old_params[param.name])
            
        self._check_param_values(self.params)
        """

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
                num_epochs = self.model.monitor.get_epochs_seen()
                cost_file = "%s/cost.csv" %(os.getcwd())
                print "Cost file written to %s" %(cost_file)
                np.savetxt(cost_file, self.recorded_cost)

                # acceptance statistics
                num_comparisons = self.num_rejects + self.num_accepts
                num_better = num_epochs - num_comparisons
                print "Number better: %d, number accepted: %d, number rejected: %d." \
                    %(num_better, self.num_accepts, self.num_rejects)
                print " Percent accepts of comparisons: %.2f" \
                    %(float(self.num_accepts)/(num_comparisons))
                
                return False


    def _check_param_values(self, params):
        """
        Helper function that checks for NaN or Inf values in the
        parameters and raises an exception if they are found.
        
        """

        for param in params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN or Inf value in " + param.name)


    def _settings_and_monitor(self):
        """
        Helper function for set up.
        Checks the parameters and sets up a monitor for the cost
        and other functions.

        """
        # set the costs
        if self.cost is None:
            self.cost = self.model.get_default_cost()

        # check parameters
        params = list(self.model.get_params())
        assert len(params) > 0
        self._check_param_values(params)
        self.params = params

        # set up monitor
        self.monitor = Monitor.get_monitor(self.model)
        self.monitor._sanity_check()

        # test if force batch size and batch size
        if getattr(self.model, "force_batch_size", False) and \
           any(self.dataset.get_design_matrix().shape[0] % \
                   self.batch_size != 0 for self.dataset in \
                   self.monitoring_dataset.values()) and \
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
        fixed_var_descr = self.cost.get_fixed_var_descr(self.model, \
                                                            nested_args)
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



    def _set_up_sample_matrices(self):
        """
        Function to set up the dictionary of "sample" matrices.
        
        Sample matrices contain 1s and 0s and specify the entries of
        the weight matrix to update during each epoch (1 indicates
        update, 0 indicates no update). There is one sample matrix per
        parameter and it has the same shape as the weight matrix for
        that parameter. The matrices are stored together in the
        "sample_matrix_dict".
        
        The sample matrices are generated by randomly selecting a user
        specified number or fraction of the indices in each weight
        matrix. 1s are placed in the sample matrix at the locations
        corresponding to these indices, and indicate these elements of
        the weight matrix will be updated. The rest of the sample
        matrix is filled with 0s indicating these elements will not be
        updated.

        """

        self.sample_matrix_dict = dict()
        for param in self.params:
            value = param.get_value(borrow=True)
            
            # find the number of entries in the rows and columns
            # of the weight matrix associated with the parameter
            assert(len(value.shape)>= 1)   # when would this occur?
            if (len(value.shape)) == 1:
                cols = value.shape[0]
                rows = 1
            else:
                rows = value.shape[0]
                cols = value.shape[1]
            num_entries = rows*cols
            
            # compile a function that returns indices for x random
            # entries of the matrix 
            #  the code simulates randomly permuting the entries and
            #  then selecting the first x of the permuted entries
            x = T.lscalar('x')   # should this be iscalar?? 
            rv_p = self.srng.permutation((1,), n=num_entries)
            z = rv_p[0][0:x]
            f = function([x], z)

            # run the function to get indices based on the info
            # provided in the YAML file
            if self.update_method == "by_num":
                indices = f(self.num_entries_to_keep)   
            elif self.update_method == "by_fraction":
                indices = f(int(self.fraction_entries*num_entries))
            else: # other cases??
                indices = f(num_entries)
            
            # create a new matrix containing only zeros
            sample_matrix = np.zeros(value.shape)
            
            # place in 1s in the new matrix at the elements
            # corresponded to the randomly generated indices, this
            # indicates these elements should be updated
            for index in indices:
                row = int(index)/cols
                col = int(index)%cols
        
                if rows == 1:
                    sample_matrix[col] = 1
                else:
                    sample_matrix[row][col] = 1

            self.sample_matrix_dict[param.name] = sample_matrix
            


    def _metropolis_accept(self):
        """
        Function to accept/reject the next step based on
        Metropolis-Hastings.
        
        If the candidate_cost is lower than the current_cost, this
        function accepts the next step. Otherwise, accepts the step 
        with probability: 
              e^((candidate_cost-current_cost)/(k*temp))

        The first step is accepted without comparison.

        """

        cost = self.monitor.channels[self.obj_channel].val_record
        
        # accept if no cost - should not occur
        if len(cost) == 0: 
            return True
        
        # accept the first step because there is no cost to compare it
        # to 
        if len(cost) == 1:
            self.energy = cost[-1]
            self.recorded_cost.append(cost[-1])
            return True


        candidate_cost = cost[-1]
        current_cost = self.energy 

        print "Candidate Cost: %.5f" %(candidate_cost)
        print "Current Cost: %.5f" %(current_cost)

        delta_cost = candidate_cost - current_cost

        # removed fuzzy inequality here because unneeded
        if delta_cost < 0:
            print "Accepted - better cost"
            self.energy = candidate_cost
            self.recorded_cost.append(self.energy)
            return True

        else:
            
            w = cmath.e**(-delta_cost/(self.k*self.temp))
            rval = random.uniform(0, 1) 
            print "w = %.5f, rval= %.3f" %(w, rval)

            if w > rval:
                print "Accepted"
                self.energy = candidate_cost
                self.recorded_cost.append(self.energy)
                self.num_accepts = self.num_accepts + 1                
                return True
  
            else:
                print "Rejected"
                self.energy = current_cost
                self.recorded_cost.append(self.energy)
                self.num_rejects = self.num_rejects+1
                return False


    def _train_params(self, params):

        # Random displacement
        old_params = dict()

        for param in params:

            param_value = param.get_value(borrow=True)

            # save a copy of the old value
            old_params[param.name]=param_value

            # use displacement of a subset of the entries to get a
            # matrix of new weight values for the parameter
            sample_matrix = self.sample_matrix_dict[param.name]
            new_value = as_floatX(self.updates_dict[param.name]\
                                      (as_floatX(param_value), sample_matrix))
            
            # updates the value of the parameter and the cost 
            param.set_value(new_value) 
            
            
        #  Accept/Reject by Metropolis
        if (not self._metropolis_accept()):
            # if reject by metropolis, set back to the value in the old_params
            for param in params:
                param.set_value(old_params[param.name])
            
        self._check_param_values(params)
