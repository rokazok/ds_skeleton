import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import yaml
import copy
import numpy as np
import pandas as pd
import joblib
import nordypy


from src.helpers.datascience_initializer import DSConfig


class AFP(DSConfig):
    """ AFP (Airflow Parallelizer) partitions data and creates a YAML with assigned workers.
    This class sets up the parallelization for workers and can be called again by 
    workers to get partitioned data. This class will be instantiated two separate times for
    two separate jobs (i.e. assign workers and get worker data), so methods for each of those
    types of jobs should consider their attributes may differ (ex. parallel_name) could
    differ depending on the job that needs it.

    Attributes
    ----------
    variables : str
        Name of key in YAML file holding variables like s3 keys and table names.
    envi : str
        Name of key in YAML file holding environment variables like prod/nonprod
        connnection strings. ex. 'prod'
    workers : int
        Number of parallel workers.
    parallel_name : str
        Name for the partition. Different names can be specified for different types of partitions.
        ex. One job might need data evenly split, another job might split data by channel, etc.
    """

    def __init__(self, variables='universal', envi=None, workers=6, parallel_name='generic'):
        super(AFP, self).__init__(variables=variables, envi=envi)

        self.n_workers = workers  # Number of workers across which to parallelize a job
        self.parallel_name = parallel_name

        # S3 key(s)
        self.s3k_parallel = self.vars['s3_key']['worker_assignment']   # where to write worker assignments

    ##################################################################################
    ##  SETUP: before parallelization, these methods split data and assign workers  ##
    ##################################################################################

    def _partition_data(self, df, n_workers=None, filename=None, groups=['dept_num', 'class_num']):
        """ Partition grouped data across workers. All data within a group are assigned 
        to a worker so that data are not split across workers.

        Dependencies
        ------------
        n_workers, s3, s3k_parallel, parallel_name

        Parameters
        ----------
        df : pandas.DataFrame
            Data to partition.
        n_workers : int
            Number of parallel workers to use.
        filename : str
            Name of file in S3 with parallel assignments
        groups : list(str)
            Grouping columns to partition. 

        Returns
        -------
        s3://{self.s3}/{self.s3k_parallel}/{self.parallel_name}.csv

        Examples
        --------
        self._partition_data(df, n_workers=20, filename='splitDataOver20workers', groups=['dept_num', 'class_num'])
        self._partition_data(df, n_workers=None, filename='splitDataByChnl', groups=['channel_num'])
        """
        filename = filename or self.parallel_name

        grp = df.loc[:,groups].drop_duplicates()

        if n_workers is None:
            # Set number of workers based on groups
            # (Make sure you do not have too many groups)
            n_workers = grp.shape[0]

        n_row = grp.shape[0]
        grp['worker'] = 1 + np.arange(n_row)//(n_row/n_workers) 

        # Save to S3
        nordypy.pandas_to_s3(data=grp,
                             bucket=self.s3,
                             s3_filepath=f'{self.s3k_parallel}/{filename}.csv',
                             header=True,
                             delimiter=',')

    def _create_schedule(self, template='parallel_worker_template', output_file=None, taskname=None, workers=None):
        """ Read in a template YAML and created a parallelized job.
        This function is manually run once to set up your job:
        1. In a dev environment, copy the template then customize the jobs.
           Do not change the names PARALLELJOB inside the yaml.
        2. Call this class with the number of workers desired and job name.
        3. Run this function to create your schedule YAML.
        
        The text PARALLELJOB tells this function which job to parallelize and
        lists those parallel jobs as dependencies for the next downstream job.
        See the schedule/template/parallel_worker_template.yaml for more notes.

        Make sure you set your ENV variables in your template. ex. if you are splitting
        a task across 9 workers, with 3 running at a time (see examples #2-3), your template yaml might look like (some fields omitted):
        cron: "0 0 * * 0"
        iam_arn: "arn:aws:iam::123456789:role/k8s/my_iam_role"
        tasks:
          - name: upstream
            path-to-executable: python dataprep.py
          - name: PARALLELJOB1
            path-to-executable: python model_training1.py
            depends-on: [upstream]
            variables:
              - name: WORKER_ID
              value: '1'
          - name: PARALLELJOB2
            path-to-executable: python another_model.py
            depends-on: [PARALLELJOB1]
            variables:
              - name: WORKER_ID
                value: '4'
          - name: PARALLELJOB3
            path-to-executable: python scoring.py
            depends-on: [PARALLELJOB2]
            variables:
              - name: WORKER_ID
                value: '7'
        
        Dependencies
        ------------
        n_workers, parallel_name, basepath

        Parameters
        ----------
        template : str
            Name of input template file in schedule/template/{template}.yaml
            One example exists in the folder, and other files in the folder are git-ignored.
        output_file : str
            Name of output yaml in schedule/autonomous/{output_file}.yaml.
            If None, use self.parallel_name.
        taskname : list(str)
            What to call the parallel tasks in the schedule yaml.
        workers : list(int)
            How many workers to assign to each parallelized task.
            If len(workers) < len(taskname), then fill in with self.n_workers
            If any workers are None, then make the same number of workers as the upstream parallel job
            with all dependies on the corresponding upstream job. 
            ex. current1 depends on upstream1, current2 on upstream2, ... 

        Returns
        -------
        schedule/autonomous/{self.parallel_name}.yaml
            Schedule yaml with parallelized jobs.

        Examples
        --------
        self._create_schedule(template='parallel_worker_template', taskname=['cooljob'], workers=[2])
        #  Ex. 1.
        #           / cooljob1 \
        #  upstream             downstream
        #           \ cooljob2 /
        
        
        self._create_schedule(template='template', taskname=['foo-','bar-','hellow-'], workers=[3,None,None])
        # Ex 2. PARALLEL-SERIAL jobs (split a large single task)
        #
        #           / foo-1 - bar-1 - hellow-1 \
        #  upstream - foo-2 - bar-2 - hellow-2 - downstream
        #           \ foo-3 - bar-3 - hellow-3 /
        

        self._create_schedule(template='template', taskname=['foo-','bar-','hellow-'], workers=[3,3,3])
        # Ex 3. Fully PARALLEL jobs (multiple, independent parallelized tasks dependent on all previous jobs finishing.)
        #
        #           / foo-1 \   / bar-1 \   / hellow-1 \
        #  upstream - foo-2  ---  bar-2  ---  hellow-2 - downstream
        #           \ foo-3 /   \ bar-3 /   \ hellow-3 /
        

        self._create_schedule(template='template', taskname=['trainA-','trainB-','score-'], workers=[3,None,2])
        # Ex 4. Larger PARALLEL-SERIAL job followed by smaller PARALLEL job
        #
        #           / traina-1 - trainb-1 \   / score-1 \
        #  upstream - traina-2 - trainb-2  ---           - downstream
        #           \ traina-3 - trainb-3 /   \ score-2 /

        """
        # Use Config._get_config() to read template schedule yaml.
        template = self._get_config(file=f'schedule/template/{template}.yaml', all_keys=True)
        tasks = template.pop('tasks')  # pull tasks from yaml

        def check_job_name(root='parallel-job', limit=15, workers=0):
            """ Truncate parallel job name to character-limit.

            Dependencies
            ------------
            n_workers, LOG

            Parameters
            ----------
            root : str
                parallel job name
            limit : int
                character limit. Default = 15.
            workers : int
                Number of workers, where max workers is used enforce char limit. ex. 100 = 3 chars.
            """
            max_len_workers=0
            if workers:
                max_len_workers = len(str(workers))
            job_nchars = len(root) + max_len_workers
            root_shortened = root  # placeholder
            if job_nchars > limit:
                root_shortened = root[:limit-job_nchars]
                self.LOG.warning(f"Parallel job name: {root} (max chars {job_nchars}) truncated to {root_shortened}.")   
            return root_shortened
        K8S_JOB_CHAR_LIMIT=15
        # if user did not supply a list of tasknames, use parallel_name0-, parallel_name1-, etc...
        n_parallel = len([t['name'] for t in tasks if 'PARALLELJOB' in t['name']])
        if not taskname:
            if n_parallel > 1:
                taskname = [f'{self.parallel_name}{i}-' for i in range(n_parallel)]
            else:
                taskname = [self.parallel_name]
        
        # workers should be a list. If none, create a list from default self.n_workers.
        workers = workers or [self.n_workers]
        # if workers list is less than number of parallel tasks, fill in with self.n_workers
        if len(workers) < n_parallel:
            workers.extend([self.n_workers]*(n_parallel-len(workers)))
        # Fill in None for parallel-serial jobs so check_job_name() does not bug with root lengths at k8s max (15 chars)
        workers_char_check = workers.copy()
        for i,e in enumerate(workers[:-1], 1):
            if workers_char_check[i] is None:
                workers_char_check[i] = workers_char_check[i-1]

        # check for valid task names within char limits
        taskname = [check_job_name(root=tn.lower().replace('_','-'), workers=workers_char_check[i], limit=K8S_JOB_CHAR_LIMIT) for i,tn in enumerate(taskname)]

        # Step through tasks and make edits as necessary
        # Serial tasks = no change
        # Parallel tasks = replicate by n_workers
        # Parallel-dependent tasks = add parallel dependencies
        revised_tasks = []  # output new yaml tasks
        generic_parallel = {}  # hold job names for dependencies
        n_parallel = 0  # count parallel jobs
        fl_serial_parallel = {} # dict to record special-case of serially-parallel jobs

        for t in range(len(tasks)):
            name = tasks[t]['name']  # store the generic PARALLELJOB root for a dict() key

            # If task is parallelized, replicate the task x workers
            if 'PARALLELJOB' in name:
                # rename to parallel taskname root
                tasks[t]['name'] = taskname[n_parallel] 
                parallel_jobs = []
                env_worker = [v for v in tasks[t]['variables'] if 'WORKER_ID' in v.values()]
                env_other = [v for v in tasks[t]['variables'] if 'WORKER_ID' not in v.values()]
                init_worker_id = self._string_parser(env_worker[0].get('value',1))
                
                # Special case: if worker[n_parallel] = None, then PARALLELJOB(X) is serial after PARALLELJOB(X-1) or next int value
                # We need to keep track of the upstream dependency job in this case
                # Figure out n_workers for this task. If none, use previous values
                worker_idx = copy.deepcopy(n_parallel)
                while workers[worker_idx] is None:
                    worker_idx -= 1
                    fl_serial_parallel = True

                # Replicate parallel job across number of workers for that job
                for i in np.arange(workers[worker_idx]):
                    wi = copy.deepcopy(tasks[t])  #  wi = worker i
                    wi['name'] = tasks[t]['name'] + str(i+1)
                    
                    # Create new WORKED_ID ENV variable for each worker, incrementing by i
                    env_worker1 = copy.deepcopy(env_worker)
                    env_worker1[0].update({'value':str(self._string_parser(init_worker_id) + i)})
                    wi['variables'] = env_worker1 + copy.deepcopy(env_other) # deep copy to sidestep yaml anchor aliases

                    # Add new task name to list of parallel tasks
                    parallel_jobs.append(wi['name'])

                    if fl_serial_parallel:
                        wi['depends-on'] = [taskname[n_parallel-1] + str(i+1)]

                    # Add task to output
                    revised_tasks.append(wi)

                # Add parallel tasks as key-value pair so dependencies can look them up.
                generic_parallel.update({name:parallel_jobs})

                # increment counters and reset flags for next task
                n_parallel += 1
                fl_serial_parallel = False
            
            else:
                # Add non-parallel tasks in order to revised_tasks output
                revised_tasks.append(tasks[t])

        # Go back and update any dependencies
        for t in revised_tasks:
            if any(['PARALLELJOB' in d for d in t.get('depends-on', ['None result'])]):
                parent = t['depends-on']
                parent = [generic_parallel[p] if 'PARALLELJOB' in p else [p] for p in parent]
                parent = [item for sublist in parent for item in sublist]  # flatten list
                t.update({'depends-on': parent})

        # convert all ENV variables to str
        for t in tasks:
            try:
                [v.update({'value':str(v['value'])}) for v in t['variables']]
            except KeyError:
                pass

        revised_tasks = list(filter(None, revised_tasks))  # remove None if present 

        # Make sure job names are k8s compliant (lowercase, numbers, no underscores)
        for d in revised_tasks:
            d['name'] = check_job_name(root=d['name'].lower().replace('_','-'), limit=K8S_JOB_CHAR_LIMIT)

        # Append tasks to original template.
        template.update({'tasks':revised_tasks})

        # Save schedule YAML
        output_file = output_file or self.parallel_name.lower()
        with open(f"{self.basepath}/schedule/autonomous/{output_file}.yaml", 'w') as outfile:
            self.LOG.info(f"Writing {self.basepath}/schedule/autonomous/{output_file}.yaml")
            yaml.dump(template, outfile, default_flow_style=None, sort_keys=False, width=float("inf"))

        return template

### See lib.initializer MFPConfig() for functions to subset worker assignments based on created worker assignments


""" EXAMPLES
import os
import sys
sys.path.append(os.getcwd())

# Create a schedule for the MFProphet model.
from training.prophet import MFProphet
# Get the data
mfp = MFProphet()
mfp.data = mfp.load_ts_data()
# the load_ts_data() function is:
df = nordypy.s3_to_pandas(bucket='merch-financial-planning', s3_filepath='input/timeseries/hts.csv').reset_index()

from lib.parallelizer import AFP
# PART 1 (run once): Assign workers and create YAML. parallel_name = filename
afp = AFP(parallel_name='foobar')
# Calculate number of workers based on number of groups (here, number of distinct channels)
self._partition_data(df=mfp.data, n_workers=None, groups=['CHANNEL_IDNT'])
# Worker assignments uploaded to s3://merch-financial-planning/parallel/foobar.csv

# Create schedule yaml with N (=workers) parallel jobs
# Copy schedule/template/parallel_worker_template.yaml and update with your pipeline and model.
# Call your new template yaml with this method to create a production yaml:
afp._create_schedule(file='parallel_worker_template') # change file as needed
# This will create schedule/autonomous/foobar.yaml (filename = afp.parallel_name)


# PART 2 (runs with every job): Get data assignments for workers
# Your model should be able to run serially or filter data based on worker.
# Add MFPConfig() as a superclass to your Model class (or call MFPConfig in the run statement), ex:

class MyModel(MFPConfig):
    ... add init and class methods here
def run(self):
    # All data for serial-run
    self.data = self.get_full_data()
    # Subset data for parallelization using AFP
    # If no worker is set, model will run serially.
    self._get_worker_assignments()
    self.data = self._get_worker_data(df_full=self.data)


# git add, commit, and push the changes.
# Schedule your job in MLP Airflow


# Here is another example of partitioning data by dept across 20 workers:
df = nordypy.s3_to_pandas(bucket='merch-financial-planning', s3_filepath='input/timeseries/hts.csv').reset_index()
from lib.parallelizer import AFP
a = AFP(workers=20, parallel_name='hello')
a._partition_data(df, n_workers=20, groups=['DEPT_IDNT'])
# creates s3://merch-financial-planning/parallel/hello.csv
a._create_schedule(file='my_super_cool_template') 
# creates repo/schedule/autonomous/hello.yaml

# If you update your worker assignments manually, ex. adding more RAM to specific workers,
# you can upload them to S3 with:
import nordypy
nordypy.s3_upload(bucket=afp.s3,
                  s3_filepath=f"{afp.vars['s3_key']['worker_assignment']}/weekly_prod.csv",
                  local_filepath=f"{afp.basepath}/adhoc/weekly_prod.csv")
"""



class LocalParallelize(object):
    """ This class generalizes parallelization of functions across a single computer's cores or threads using joblib.

    Attributes
    ----------
    n_cores : int
        Number of cores or threads. -1 = use all available.
    """
    def __init__(self, cores=-1, **kwargs):
        super(LocalParallelize, self).__init__(**kwargs)
        self.cores = cores  # N cores to pass to joblib
        self.n_cores = joblib.cpu_count() # Calculated machine cores

    @ staticmethod
    def _serial_wrapper(func, func_var, iterable, **kwargs):
        """ Although this class is designed for parallelization, certain tasks may need to run serially or parallel functions need to be
        benchmarked against serial execution. This function will pass iterable to the function like: func(func_var = i, **kwargs)

        Parameters
        ----------
        df : pandas.DataFrame
            Data on which to run parallelized functions.
        groups : dict(str:str)
            Dict of grouping columns that are used to filter functions, and their corresponding function variables used for filtering.
            ex. {'CHANNEL_IDNT':'channel', 'DEPT_IDNT':'dept'}
            This dict can be created from dict(zip(['col1', 'col2'], ['var1','var2'])).
        func : function
            Function. Pass variables as kwargs.
        func_var : str
            Iterable will be passed to this function variable.
        iterable : list
            List over which to iteratively pass values to func_var.
        kwargs : dict
            func variables and their values to pass.

        Returns
        -------
        list

        """
        results = []
        for i in iterable:
            var_kwarg = dict(zip([func_var], [i]))  # package variable and iterable together to pass to func
            results.append(func(**var_kwarg, **kwargs))
        return results

    @ staticmethod
    def _partitioned_df(df_list, func, func_df='df', cores=-1, prefer='processes', **kwargs):
        """ Parallelize operations on a dataframe split into partitions. Use this parallelization when
        1) the function intakes a dataframe and subsets it then performs an operation. ex. channel_forecast(df=all_jwn_data, channel=channel)
        2) the function returns a pandas.DataFrame.

        Parameters
        ----------
        df_list : list(pandas.DataFrame)
            List of data partitions on which to run parallelized functions.
        func : function
            Desired function to parallelize. Function variables can be passed as kwargs.
            First argument of the function will be the passed
        func_df : str
            Name of the function variable corresponding to the data. ex. if myfun(df=mydf, var=myvar), then func_df='df'.
        cores : int
            Number of cores
        prefer : str ['processes', 'threads']
            Use processes or threads. See: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        kwargs : dict
            func variables and their values to pass.

        Returns
        -------
        pandas.DataFrame
        """
        executor = joblib.Parallel(n_jobs=cores, verbose=10, prefer=prefer)
        tasks = (joblib.delayed(func)(**dict(zip([func_df], [df_i])), **kwargs) for df_i in df_list)
        # tasks = (joblib.delayed(func)(df_i, **kwargs) for df_i in df_list)
        results = (executor(tasks))
        return pd.concat(results)

    @ staticmethod
    def _grouped_df(df, groups, func, cores=-1, prefer='threads', **kwargs):
        """ Parallelize operations on a single dataframe by groups. Use this parallelization when
        1) the function intakes a dataframe and subsets it then performs an operation. ex. channel_forecast(df=all_jwn_data, channel=channel)
        2) the function returns a pandas.DataFrame.
        This function subsets the single dataframe by groups.keys() then passes each subsetted group into the function as
        function(groups.values() = subset_group, ...)
        The joblib.delayed() function orchestrates this variable handling with the kwargs.

        Parameters
        ----------
        df : pandas.DataFrame
            Data on which to run parallelized functions.
        groups : dict(str:str)
            Dict of grouping columns that are used to filter functions, and their corresponding function variables used for filtering.
            ex. {'CHANNEL_IDNT':'channel', 'DEPT_IDNT':'dept'}
            This dict can be created from dict(zip(['col1', 'col2'], ['var1','var2'])).
        func : function
            Desired function to parallelize. Function variables can be passed as kwargs.
        cores : int
            Number of cores
        prefer : str ['processes', 'threads']
            Use processes or threads. See: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        kwargs : dict
            func variables and their values to pass.

        Returns
        -------
        pandas.DataFrame
        """
        df_groups = list(groups.keys())
        func_vars = list(groups.values())

        # Get array of groups.
        keys = df[groups].drop_duplicates().values

        # Create an executor
        executor = joblib.Parallel(n_jobs=cores, verbose=10, prefer=prefer)

        # Parallelize function
        # The single df is passed to each function, along with function variables corresponding to groups.
        # Additional kwargs can be passed.
        tasks = (joblib.delayed(func)(df=df, **dict(zip(func_vars, k)), **kwargs) for k in keys)

        results = (executor(tasks))
        return pd.concat(results)

    @ staticmethod
    def _parallel_over_list(fn, fn_args=None, l=None, cores=-1, prefer='threads', **kwargs):
        """ Parallelize operations on a single dataframe by groups. Use this parallelization when
        1) the function intakes a dataframe and subsets it then performs an operation. ex. channel_forecast(df=all_jwn_data, channel=channel)
        2) the function returns a pandas.DataFrame.
        The joblib.delayed() function orchestrates this variable handling with the kwargs.

        The scenarios in fn_args work like:
        1. User does not pass any args, then treat list as *args. 
        2. User passes named args. Each item in list matches to the args. Function zips them together.
           Use this when each sublist item maps to specific arguments in the function.
        3. User passes a single fn_arg. Each item in list is a for-loop for that fn_arg on a worker.
           Use this when each worker iterates through a list.

        Parameters
        ----------
        df : pandas.DataFrame
            Data on which to run parallelized functions.
        groups : dict(str:str)
            Dict of grouping columns that are used to filter functions, and their corresponding function variables used for filtering.
            ex. {'CHANNEL_IDNT':'channel', 'DEPT_IDNT':'dept'}
            This dict can be created from dict(zip(['col1', 'col2'], ['var1','var2'])).
        fn : function
            Desired function to parallelize. Function variables can be passed as kwargs.
        fn_args : list(str) or None
            Function argument(s):
            1. if fn_args is None, then list iterable is passed as *args, ex.
                if my_fn(*args), then fn_args=None
            2. if len(fn_args) == len(list iterable), then they are passed together, ex. 
                if my_fn(foo='cat', bar=2), then fn_arg=['foo','bar']
            3. if len(fn_args) == 1 and len(list iterable)>1, then the list is passed with fn_arg, ex.
                if my_fn(foo=[3,7,9]), then fn_arg=['foo'] and l=[[3,7,9], ...]
        l : list(lists or tuples)
            List of function arg values to parallelize. Each list item goes to a different worker.
            ex. if we wanted to parallelize the function above across 3 instances, 
            we could pass l=[['cat',2], ['dog',2], ['fish',4]]
        cores : int
            Number of cores
        prefer : str ['processes', 'threads']
            Use processes or threads. See: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        kwargs : dict
            more func variables and their values to pass. These can be constants.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        from lib.parallelizer import LocalParallelize
        import random, time
        def sleep_and_print(a='foo', b=1):
            "Simple sleep function. Add b seconds to default s sleep then print statements."
            #s = random.randrange(10,21)
            s=15
            print(f"a={a}, b={b}, s={s}, sleeping for b+s {b+s} sec")
            time.sleep(b+s)
            print(f"a={a} // sleep done")

        my_args = list(zip(['foo','bar','baz','hello'],[1,2,4,1]))
        # [('foo', 1), ('bar', 2), ('baz', 4), ('hello', 1)]

        # Serially run function to illustrate time savings.
        t0 = time.time()
        for i in my_args:
            sleep_and_print(**dict(zip(['a','b'],i)))
        t1 = time.time()
        print(f"==== Serial Elapsed: {t1-t0} s ====")


        self._parallel_over_list(fn=sleep_and_print, fn_args=['a','b'], l=my_args)
        t2 = time.time()
        print(f"==== Parallel Elapsed: {t2-t1} s ====")
        # ^Scenario 2 example: Each tuple myarg[i] is passed into sleep_and_print like sleep_and_print(a=myarg[i][0], b=myarg[i][1]) 
        # So the first tuple is sleep_and_print(a='foo', b=1) 

        # Scenario 1 example: If you don't pass fn_args, first tuple is passed like sleep_and_print('foo',1)
        self._parallel_over_list(fn=sleep_and_print, l=my_args)

        # Scenario 3 example: You pass a single fn_arg and l is a list
        # (Note: arg a is a constant in this example)
        self._parallel_over_list(fn=sleep_and_print, fn_args=['b'], l=[[1],[2],[3],[4]], **{'a':'hi'})
        """
        # Create an executor
        executor = joblib.Parallel(n_jobs=cores, verbose=10, prefer=prefer)

        # Parallelize function
        # Additional kwargs can be passed. see doc strings section on fn_args
        if fn_args is None:                             # Scenario 1
            tasks = (joblib.delayed(fn)(*i, **kwargs) for i in l)
        elif len(fn_args)==len(l[0]):                   # Scenario 2
            tasks = (joblib.delayed(fn)(**dict(zip(fn_args, i)), **kwargs) for i in l)
        elif (len(fn_args)==1) & (len(l[0])>1):         # Scenario 3
            tasks = (joblib.delayed(fn)(**{fn_args[0]:i}, **kwargs) for i in l)

        results = (executor(tasks))
        return results

""" EXAMPLES

# Create data
import pandas as pd
df = pd.DataFrame({'chan_id':[1,1,1,1,2,2,2,2],
                   'dept_id':[101,101,121,121,200,200,230,230],
                   'val':[11,99,22,88,33,77,44,66]})
df
>
	chan_id 	dept_id 	val
0 	1 	        101 	    11
1 	1 	        101 	    99
2 	1 	        121 	    22
3 	1 	        121 	    88
4 	2 	        200 	    33
5 	2 	        200 	    77
6 	2 	        230 	    44
7 	2 	        230 	    66

# test function: find max and use *args or *kwargs
# additional function variables:
# - filter on dept
# - add new column (col) with value (val)
def demo_max(df, dept=None, col=None, val=-1):
    if dept:
        df = df.loc[df.dept_id == dept,].copy()
    results = df.apply(max).to_frame().T
    if col:
        results[col]=val
    return results

demo_max(df)
>
	chan_id 	dept_id 	val
0 	2 	        230 	    99

myargs = {'col':'newcol', 'val':949}
demo_max(df, **myargs)            # dept=230, val=99, newcol=949
demo_max(df, **myargs, dept=121)  # dept=121, val=88, newcol=949

######################################
##  Serially iterate through depts  ##
######################################
myargs = {'df':tmp, 'col':'newcol', 'val':949}
r = LocalParallelize._serial_wrapper(func=demo_max, func_var='dept', iterable=[101,121,200,230], **myargs)
pd.concat(r)
>
 	chan_id 	dept_id 	val 	newcol
0 	1 	        101 	    99 	    949
0 	1 	        121 	    88 	    949
0 	2 	        200 	    77 	    949
0 	2 	        230 	    66 	    949

###########################
##  Parallelize by dept  ##
###########################
grps = dict(zip(['dept_id'],['dept']))
LocalParallelize._grouped_df(groups=grps, func=demo_max, prefer='processes', **myargs)
>
    chan_id 	dept_id 	val 	newcol
0 	1 	        101 	    99 	    949
0 	1 	        121 	    88 	    949
0 	2 	        200 	    77 	    949
0 	2 	        230 	    66 	    949

#################################
##  Parallelize on partitions  ##
#################################
# Prepare partitions
gb = df.groupby(['chan_id', 'dept_id'])
df_l = [gb.get_group(x) for x in gb.groups]
df_l
>
[   chan_id  dept_id  val
 0        1      101   11
 1        1      101   99,
    chan_id  dept_id  val
 2        1      121   22
 3        1      121   88,
    chan_id  dept_id  val
 4        2      200   33
 5        2      200   77,
    chan_id  dept_id  val
 6        2      230   44
 7        2      230   66]

myargs1 = {'col':'newcol', 'val':949}
LocalParallelize._partitioned_df(df_list=df_l, func=demo_max, func_df='df', **myargs1)
>
 	chan_id 	dept_id 	val 	newcol
0 	1 	        101 	    99 	    949
0 	1 	        121 	    88 	    949
0 	2 	        200 	    77 	    949
0 	2 	        230 	    66 	    949

# Parallel operations: see doc-strings for
"""
