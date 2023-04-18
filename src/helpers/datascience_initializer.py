import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
import datetime
import functools
import numpy as np
import pandas as pd
import operator
from ast import literal_eval

import boto3
from botocore.exceptions import ClientError
import awswrangler
from cerebro.connection_store import Client
import nordypy

from src.helpers.initializer import Config

class DSConfig(Config):
    """ Set up configuration and methods relevant to data science projects, including setting forecast fiscal week.
    This class imports the general Config, and creates environment connections for nordypy.

    How to use it with your custom MySubclass:
    class MySubclass(DSConfig):
        def __init__(self, variables='universal', envi=None, fcst_dt=None, parallel_name=None):
            super().__init__(variables=variables, envi=envi, fcst_dt=fcst_dt, parallel_name=parallel_name)
            # __init__ passes variables from MySubclass to DSConfig superclass
            # Rest of your code here.

    Attributes
    ----------
    variables : str
        Name of key in config/config.YAML file holding variables like s3 keys and table names.
    envi : str
        Environment, like 'nonprod', 'prod', or your LANID.
        This is used to set self.ENVIRONMENT (for connection keys) and 
        self.ENVI_TYPE (for noting if we're running in prod or nonprod. Useful for setting LDAP='auth' for teradata connections)
    parallel_name: str
        Parallel assignments file found in: f"{self.vars['s3_key']['worker_assignment']}/{parallel_name}.csv"
        This explicit argument is useful if there are different parallel assignments for the same model, 
        ex. hyperparameter tuning on 20 workers and model training on 8 workers.
    fcst_wk : int (or str)
        First fiscal week of forecast as YYYYWW, ex. 202043.
    db_creds : bool
        Should the instance retrieve MLP creds for nordypy connections?
        True for most applications. False for API.
    ref_data : bool
        Should the instance retrieve reference data?
        True for most applications. False for speed, API.

    Returns self.* Attributes
    -------------------------
    s3 : str
        MFP s3 bucket
    RUN_DATE_UTC/LOCAL : str
        Date the code was run as 'YYYY-MM-DD'.
        Convert back to datetime.date with datetime.datetime.strptime(RUN_DATE_UTC, '%Y-%m-%d').date()
    dates : pandas.DataFrame
        Fiscal date and week lookup from NAP. See self._get_dates()
    fcst_dt : str
        Date of forecast as YYYY-MM-DD. If no date entered, find start of current fiscal week (Sunday) based on current date.
    """

    def __init__(self, variables='universal', envi=None, parallel_name=None, fcst_dt=None, db_creds=True, ref_data=True):
        super(DSConfig, self).__init__(variables=variables, envi=envi)
        # Initialization
        Path(f"{self.basepath}/data/").mkdir(parents=True, exist_ok=True) # make data/ directory if not exists
        self.s3 = self.vars['s3']     # PIF DS bucket
        self.RUN_DATE_UTC = str(datetime.datetime.utcnow().date()) # UTC date this script was run
        self.RUN_DATE_LOCAL = str(datetime.datetime.now().date())  # Local date this script was run
        # Note: convert str back to datetime.date with datetime.date.fromisoformat(self.RUN_DATE_UTC)

        # Update args from s3, ex. set a past forecast week for for backcasting
        if self.args.args_from_s3:
            self._load_args_from_s3(file=self.args.args_from_s3)

        # Set parallel_name from object initiation or parsed arg
        self.parallel_name = parallel_name or self.args.parallel_name

        # Convert MLP connection strings to nordpy ENV connections
        if db_creds:
            self._nordypy_creds()

        # Load a hard copy of dates
        if ref_data:
            # Get a lookup of dates and fiscal week
            # since refresh=False, this class initializes with a copy saved to S3
            self.dates = self._get_reference_data(conn=self.conn['nap'], file='dates', 
                                                s3_key=self.vars['s3_key']['reference'],
                                                sql_params={'date':self.RUN_DATE_UTC, 'future_yrs':10}) 
            self.hierarchy = self._get_reference_data(conn=self.conn['nap'], file='merch_hierarchy', s3_key=self.vars['s3_key']['reference'])
            self.dma = self._get_reference_data(conn=self.conn['nap'], file='dma', s3_key=self.vars['s3_key']['reference'])

        # Set forecast date to beginning of fiscal week. If no date, use current date. 
        tmp_fcst_dt = fcst_dt or self.args.fcst_dt or self.RUN_DATE_LOCAL
        if hasattr(self, 'dates'):
            self.fcst_wk = self._date_to_week_lookup(dates=self.dates, date_lkp=tmp_fcst_dt, date_col='week')
            self.fcst_dt = self._date_to_week_lookup(dates=self.dates, date_lkp=self.fcst_wk)  # reset to start of fiscal week (Sunday)

    ###########################################################
    ##  PARALLELIZATION: For each worker, get assignments.   ##
    ###########################################################

    def _get_worker_assignments(self, parallel_name=None, proceed=False):
        """ Get worker assignments from S3.
        Columns are grouping columns + worker, where grouping columns are inner-joined to your
        data to subset to different workers, and worker is set in your schedule DAG as an ENV variable
        that matches the 'worker':ENV_variable registered in config/env_variables.yaml

        ex. parallel_name.CSV
        channel | dept | ... | worker 
        --------|------|-----|-------
        N.COM   | 800  | ... |  1
        N.COM   | 850  | ... |  2
        FLS     | 800  | ... |  3
        FLS     | 850  | ... |  4

        Attribute Paramters
        -------------------
        s3, vars, LOG

        Parameters
        ----------
        parallel_name : str
            Name of CSV file in S3 parallel key folder containing worker assignments.
        proceed : bool
            Proceed with all data if worker assignment file is not found?
            Not recommended because all parallel jobs will run with full data.
            Setting this flag to True is useful for local troubleshooting.

        Returns
        -------
        pandas.DataFrame
            Dataframe with grouping columns and workers.
        """
        if not parallel_name:
            return None
        
        key = f"{self.vars['s3_key']['worker_assignment']}/{parallel_name}.csv"

        try:
            assignments = nordypy.s3_to_pandas(bucket=self.s3,
                                    s3_filepath=key,
                                    index_col=False, header=0)
            return assignments
        except ClientError as e:
            self.LOG.exception(f"NoSuchKey: s3://{self.s3}/{key}")
            if proceed:
                self.LOG.warning("No worker assignments, continue with all data.")
                pass

    def _get_worker_data(self, df_full, assignments):
        """ Partition the full dataset. First, select the groups assigneded to this worker, 
        where this worker is read from the ENV variable "WORKER_ID" or whatever the ENV variable is named in config/env_variables.yaml.
        Then inner-join those groups with the full data to get the subset data.
        If no worker is defined, take all data (useful for redundancy, i.e. serial run of full data).

        Attribute Parameters
        --------------------
        ENV, assignments

        Parameters
        ----------
        df_full : pandas.DataFrame
            Full dataset to partition. Must not have a column called 'worker' so the inner join works.

        assignments : pandas.DataFrame
            Dataframe with grouping variables and worker ID (as column 'worker') to inner join on df_full.

        Returns
        -------
        pandas.DataFrame
            Subset data based on inner-join of worker data
        """
        if not isinstance(assignments, pd.DataFrame):
            self.LOG.info("No assignments to load. Returning full data")
            return df_full

        worker = self.ENV['worker']
        if worker:
            # Subset groups by worker.
            grp = assignments.loc[assignments.worker == worker].drop(columns='worker')
        else:
            self.LOG.info("No WORKER_ID ENV variable detected.")
            return df_full

        # Inner join full data to worker's groups.
        df_subset = df_full.merge(grp, how='inner')

        return df_subset

    ###########################################################
    ##  PARALLELIZATION v2                                   ##
    ###########################################################

    def _setup_worker_assignments(self, s3_bucket, s3_folder, file_name, n_workers, df, size_col=None):
        """
        Assigns each task to a worker and pushes worker assignments csv to S3. 
        This function workers with the setup_worker() function below. 
        If there is a size column in the dataframe, the final worker assignments should have similar total sizes. 
        If there are no rows in the given df, add a row of zeros to assign to one worker. This will be taken care of in the setup_worker() function below.  

        Args:
            s3_bucket (str): S3 bucket
            s3_folder (str): S3 folder
            file_name (str): S3 file name
            n_workers (int): number of workers used to split all tasks
            df (dataframe): dataframe with pregrouped columns. length of df is number of tasks. 
            size_col (str): name of column in df that includes size information, or None
        Returns:
            dataframe: original df with worker assignment. each row will have workers assigned, and workers will typically run multiple models
        """
        # sort largest to smallest if there is a size column
        if size_col is not None: 
            df = df.sort_values(size_col, ascending=False).reset_index(drop=True)

        # if there are no rows in the given df, add a row of zeros 
        if len(df) == 0: 
            df = df.append({k:0 for k in df.columns}, ignore_index=True)

        # n_workers is the minimum of 1. the number of groups and 2. the inputed number of workers
        n_workers = min(len(df), n_workers)

        # n_repeat is max tasks per worker
        n_repeat = int(len(df) / n_workers) + (len(df) % n_workers > 0)

        # base_list is list of workers
        base_list = list(range(1, (n_workers + 1)))
        workers = []
        for i in range(n_repeat):
            if i % 2 == 0:
                workers += base_list
            else:
                workers += base_list[::-1]

        # add worker to original df
        df['worker'] = workers[0:len(df)]

        # shuffle df to have random ordering
        df = df.sample(frac=1, random_state=2022).reset_index(drop=True)
        
        # push worker assignments to S3
        nordypy.pandas_to_s3(
            data=df,
            bucket=s3_bucket,
            s3_filepath=f'{s3_folder}/{file_name}',
            delimiter=',',
            index=False,
            header=True)
        
        return df

    def _setup_worker(self, s3_bucket, s3_folder, file_name, worker, task_cols):
        """
        Pull worker assigments from S3 and return task inputs.
        This function workers with the setup_worker_assignments() function above. 
        
        Args:
            s3_bucket (str): S3 bucket
            s3_folder (str): S3 folder
            file_name (str): S3 file name
            worker (int): worker number
            task_cols (list): list of column names in dataframe that are task inputs; ex:['dept_idnt', 'chnl_nmbr']
        Returns:
            list: list of dictionaries of task inputs
        """
        # pull worker assignments from S3
        df = nordypy.s3_to_pandas(
            bucket=s3_bucket,
            s3_filepath=f'{s3_folder}/{file_name}',
            index_col=False,
            header=0)
    
        # get dictionary of tasks assigned to given worker
        worker = int(worker)
        worker_tasks = df[df.worker == worker][task_cols].to_dict('records')

        # check if worker tasks should actually be empty if not already empty
        # if all values are 0s, it means there are no tasks assigned to this worker
        # first check if any tasks are not numeric, then check for sum to 0
        if len(worker_tasks)>0:
            if False not in [(isinstance(v, int) or isinstance(v, float)) for k,v in worker_tasks[0].items()]: 
                if sum([sum([v for k,v in d.items()]) for d in worker_tasks]) == 0: 
                    worker_tasks = []

        return worker_tasks

    ###############################################################
    ##  DATABASE and S3 functions: Get data from S3 or database  ##
    ###############################################################

    @staticmethod
    def _cerebro_to_env(database_key):
        """
        Uses cerebro to create bash connection strings that can be accessed by nordypy.
        
        Parameters
        ----------
        database_key : string
            cerebro connection ids 

        Examples
        --------
        _cerebro_to_env('my-cerebro-key')
        """
        db = {}
        conn_client = Client()
        conn_obj = conn_client.get_connection(connection_id=database_key)
        try:
            db[database_key] = {
                'host': conn_obj.host,
                'password': conn_obj.password,
                'user': conn_obj.username,
                'port': getattr(conn_obj, 'port', None),
                'dbname': getattr(conn_obj, 'schema', None),
            }
            if hasattr(conn_obj, 'extras'):
                try:
                    extras = literal_eval(conn_obj.extras.replace('true', 'True').replace('false', 'False'))
                    for k, v in extras.items():
                        db[database_key][k] = v
                except SyntaxError:  # no actual extras
                    pass
        except KeyError:
            pass
        
        # write connection to bash
        s = ''
        for k, v in db[database_key].items():
            s += k + '=' + str(v) + ' '
        os.environ[database_key] = s
        
    def _nordypy_creds(self, connections=None, refresh=False):
        """ Create ENV variable connection strings from MLP connection store strings.
        This for-loop detects envi type (prod, nonprod, LANID), with connections stored in self.conn,
        and an adapted version of Aaron's cerebro_to_env() function to transfer those credentials into ENV variables.

        Attribute Parameters
        --------------------
        conn, LOG

        Parameters
        ----------
        connections : list(str)
            List of MLP connection strings. If None, use self.conn.
            For most purposes, this variable should be None, but it allows users
            to directly pass a list connection strings to cerebro_to_env().
        refresh : bool
            Update all keys? (True) or only missing keys (False)

        Examples
        --------
        _nordypy_creds()   # Add current environment creds as ENV variables 

        # Manually refresh creds in current session, ex. if password changed.
        _nordypy_creds(connections=['my-scno', 'my-nap'], refresh=True)
        """
        # Get all listed keys
        keys = connections or list(filter(None, self.conn.values()))

        # If refresh=False, we will only update missing keys
        if refresh is False:
            # Which keys do not exist as ENV variables?
            keys = [i for i in keys if i not in os.environ]

        if keys:
            self.LOG.info(f"Create ENV db connections for {keys}")
            for dbkey in keys:
                self._cerebro_to_env(database_key=dbkey)

    def _load_args_from_s3(self, file=None, replace=False):
        """ Load parsed args from S3 and pass them into the class as if they were called when the script is run.
        This is useful for backcasting without updating a DAG with args.

        A good way to utilize this capability is to have:
        1. the normal scheduled DAG that does NOT call the args_from_s3 parsed arg
        2. a non-scheduled, manually-triggered DAG includes the args_from_s3 flag with every python script call, 
           ex. python myscript.py --args_from_s3

        See config/config.yaml for the location of where to save parsed_args.csv in s3.
        The parsed_args.csv should have 2 columns: parsed_arg | value

        Attribute Parameters
        --------------------
        args, vars, LOG, _make_argparser

        Parameters
        ----------
        file : str
            Name of CSV file in S3 where column 1= arg name, column 2= value.
        replace : bool
            Replace all args with those in the file?
            If False, original parsed args remain unless they're replaced by file.

        Examples
        --------
        # Check if the s3 args boolean flag is true, then load args
        print(self.args)  # BEFORE
        if self.args_from_s3:
            self._load_args_from_s3(file=self.args.args_from_s3)
        print(self.args)  # AFTER
        """
        s3_key = f"s3://{self.vars['s3']}/{self.vars['s3_key']['parsed_args']}/{file}.csv"

        self.LOG.info(f"Load parsed args from {s3_key}")
        s3_args = awswrangler.s3.read_csv(path=s3_key, path_suffix=['csv'])
        # Convert data to dict.
        args_dict= dict(zip(s3_args['parsed_arg'], s3_args['value']))
        # Convert dict to argparse compatible
        arg_list=[]  # args is a list as ['--parsed_arg1', 'val1', '--arg2', 'val2']
        for k,v in args_dict.items():
            arg_list.extend(['--'+k,v])
        parser = self._make_argparser()
        self.LOG.info(f"Original args: {self.args}")
        if replace:
            self.args, unknowns_list = parser.parse_known_args(args=arg_list, namespace=None)
        else:
            _, unknowns_list = parser.parse_known_args(args=arg_list, namespace=self.args)
        self.LOG.info(f"Updated args from s3: {self.args}")

    def _partitions_to_path(self, partitions):
        """ Turn a dict of key:value pairs into a path for S3. The order of the dict is important.
        This helper function is used to delete, read, and upload files in S3, including partioned Parquet files.
        NOTE: This function assumes dict order is preserved. If order is not preserved in the future, use collections.OrderedDict

        Attributes
        ----------
        partitions : dict
            Partitions and values in order.

        Returns
        -------
        str

        Examples
        --------
        _partitions_to_path(**{'forecast_wk':202032, 'model':'myModel'})
        > 'forecast_wk=202032/model=myModel'
        _partitions_to_path(**{'model':'myModel', 'forecast_wk':202032}) # order matters
        > 'model=myModel/forecast_wk=202032'
        _partitions_to_path(**{'model':'myModel', 'forecast_wk':''}) # blank values are OK
        > 'model=myModel/forecast_wk'
        """
        path=''
        if partitions:
            for partition, partition_value in partitions.items():
                path = path + self._blank_str_check(partition, delim_before='/') + self._blank_str_check(str(partition_value), delim_before='=')
        return path.replace('/', '', 1) # replace first backslash

    def _clear_s3(self, bucket, key=None, partitions=None):
        """ Deletes partitioned files in S3. In Presto, DROP TABLE does not delete the data in S3,
        so this function is a companion to those drop table statements.
        This function can delete entire key contents or by single partition, but not a subset of partitions.
        Note: aws-okta or other authentication service needs to be running.

        Attribute Parameters
        --------------------
        LOG, _blank_str_check, _partitions_to_path

        Parameters
        ----------
        bucket : str
            S3 bucket
        key : str
            S3 key containing target for deletion.
            If key is '', then key path can be built with partitions arg.
            Default is None so that users have to intently pass '' to use this method.
        partitions: dict
            Partition and the id(s) to delete. Leave blank to delete everything in that key.
            Ex. {'year':2020, 'model':'foo', variables:''}
            s3://my-bucket/my-key/year=2020/model=foo/variables

        Returns
        -------
        None

        Examples
        --------
        # delete all contents of s3://my-bucket/my-key
        _clear_s3(bucket='my-bucket', key='my-key')

        # delete all contents of: s3://my-bucket/my-key/year=2020
        _clear_s3(bucket='my-bucket', key='my-key', partitions={'year':'2020'})

        # delete s3://my-bucket/path/to/key=99
        _clear_s3(bucket='my-bucket', partitions={'path':'', 'to':'', 'key:'99'})
        """
        bucket = bucket or self.s3
        s3 = boto3.resource('s3')
        s3bucket = s3.Bucket(bucket)
        # Key path check to avoid deleting entire bucket contents. Entire bucket can still be deleted if key='' and partitions={'':''}.
        if (key is None) or ((key=='') and (partitions is None)):
            raise Exception("Key or partition path is required.")

        delete_target = f"{self._blank_str_check(key, delim_after='/')}{self._blank_str_check(self._partitions_to_path(partitions), delim_after='/')}"
        # note: extra backslash prevents deleting similar keys, ex. path/to/reg_tree and path/to/reg_tree_boost
        self.LOG.info(f"Delete contents of s3://{bucket}/{delete_target}")
        s3bucket.objects.filter(Prefix=delete_target).delete()

    def _upload(self, df=None, bucket=None, key=None, partitions=None, overwrite=False, parquet_kwargs={}):
        """ Send pandas.DataFrame results to S3 in partitioned keys or as a single CSV. Requires authentication service.
        This function allows one to upload a single model (adding to existing results) or all model results (clearing out previous data before upload).
        See examples below, including a special case for uploading several models in a single df.

        Attribute Parameters
        --------------------
        LOG, s3

        Parameters
        ----------
        df : pandas.DataFrame
            Data to upload. 
        bucket, key : str
            S3 bucket and key prefix, ex. my-bucket/path/to/key
        partitions : dict
            Partitions for data, ex. {'forecast_wk':202115, 'model':'baseline'} 
            Keys are used to partition uploaded data.
            Key-value pairs are used to clear s3 if overwrite=True.
            Note: If partitions are last columns, Hive tables can be created on data in S3.
        overwrite : bool
            If True, delete previous results with the same (partition) key values.
            For parallelized operations, set this to False.
        parquet_kwargs : dict
            Dict of args to pass to awswrangler.s3.to_parquet().
            Useful args include filename_prefix, mode.
            See: https://aws-data-wrangler.readthedocs.io/en/stable/stubs/awswrangler.s3.to_parquet.html

        Returns
        -------
        None
        Data in S3

        Examples
        --------
        # Upload a SINGLE model results for a given week (two options)
        self._upload(df=df, bucket=self.s3, key='output/forecast', partitions={'forecast_wk':self.fcst_wk, 'model':self.model}, overwrite=True)  # overwrite results
        self._upload(df=df, bucket=self.s3, key='output/forecast', partitions={'forecast_wk':self.fcst_wk, 'model':self.model}, overwrite=False) # ideal for parallelized operations

        # Delete ALL existing model results and upload SINGLE or MULTIPLE model results for a given week 
        # (Note: since overwrite=True and model is blank, all models are deleted; for uploads, only the keys are used for partitioning data)
        self._upload(df=self.forecast, bucket=self.s3, key='output/forecast', partitions={'forecast_wk':self.fcst_wk, 'model':''}, overwrite=True)

        # Delete SPECIFIC MUTLIPLE models then upload a df with MULTIPLE models, ex. when they are concatenated together, without clearing other models. 
        # This requires a for-loop + _clear_s3(), then _upload()
        for mdl in df['model'].unique():
            # Clear out specific models in df (without deleting other models)
            self._clear_s3(bucket=self.s3, key='output/forecast', partitions={'forecast_wk':self.fcst_wk, 'model':mdl})
        self._upload(df=df, bucket=self.s3, key='output/forecast', partitions={'forecast_wk':self.fcst_wk, 'model':self.model}, overwrite=False)
        """
        bucket = bucket or self.s3  # default to project s3 bucket if None.
        s3_path = f's3://{bucket}/{key}'
        s3_kwargs = {"ACL": "bucket-owner-full-control"}

        if overwrite:
            self._clear_s3(bucket=bucket, key=key, partitions=partitions)

        if partitions:
            awswrangler.s3.to_parquet(df=df,
                                      path=s3_path,
                                      dataset=True,
                                      s3_additional_kwargs=s3_kwargs,
                                      partition_cols=list(partitions.keys()),
                                      **parquet_kwargs
                                      )
        elif key.endswith('.csv'):
            awswrangler.s3.to_csv(df=df, path=s3_path, index=False, s3_additional_kwargs=s3_kwargs)
        self.LOG.info(f"Uploaded data ({df.shape}) to {s3_path}")

    def _read_parquet(self, bucket=None, key=None, partitions=None, prefix=None):
        """ Helper wrapper function to simplify reading partitioned parquet data from S3. 
        (For specific files, just use the original awswrangler.s3.read_parquet() ).
        https://aws-data-wrangler.readthedocs.io/en/stable/stubs/awswrangler.s3.read_parquet.html
        aws-okta or other authentication service needs to be running.

        Parameters
        ----------
        bucket, key : str
            S3 bucket and key for results. 
            Key should not have a backslash at the end, as this is added automatically.
            Defaults to MFP forecasts.
        partitions : dict
            Partitions and values in order.
        partition_cols : bool
            If True, write partition dict into columns, ex.
            col 'key1' with rows = value1, col 'key2' with rows = value2, ...
        prefix : str
            Prefix for the  Parquet file.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        df = self._read_parquet(bucket='mybucket', key='path/to/key', partitions={'dept':123, 'model':'MyModel'})

        KEY="output/forecast/staged"
        DEPT=871
        df = self._read_parquet(bucket=self.s3, key=KEY, partitions={'dept':DEPT})  # loads all Parquet files
        df = self._read_parquet(bucket=self.s3, key=KEY, partitions={'dept':DEPT}, prefix="RCOM") # loads RCOM*.parquet with dept (int) column
        df = self._read_parquet(bucket=self.s3, key=f"{KEY}/dept={DEPT}", prefix="RCOM")  # loads RCOM*.parquet without dept column
        #df = self._read_parquet(bucket=self.s3, key=f"{KEY}/dept={DEPT}/RCOM_")  # Fails because forwardslash added to end of key
        """
        # set defaults if not specified
        bucket = bucket or self.s3
        key = key or self.vars['s3_key']['forecasts']
        partition_path = self._blank_str_check(self._partitions_to_path(partitions), delim_after='/')
        path = f"s3://{bucket}/{self._blank_str_check(key, delim_after='/')}{partition_path}{self._blank_str_check(prefix)}"

        if not prefix:
            df = awswrangler.s3.read_parquet(path=path, dataset=True)
        else:
            df = awswrangler.s3.read_parquet(path=path, dataset=False)
            if partitions:
                # Add partitions to non-dataset Parquet and infer datatype
                for k,v in partitions.items():
                    df[k]=v
                df = df.astype({k:type(self._string_parser(v)) for k,v in partitions.items()})
        self.LOG.info(f"Pulled data ({df.shape}) from {path}")
        df = self._string_to_datetime(df=df)
        return df

    ###########################################################################
    ##  DATA functions: modifying data. Some functions specific to merch DS  ##
    ###########################################################################
    
    def _get_reference_data(self, s3_key, conn, file, sql_params={}, refresh=False, upload=False):
        """ Get reference data, like:
        dates = fiscal calendar dates and weeks
        hierarchy = merchandise hierarchy from division down to subclass

        Search for data in this order, ex. for dates:
        1. Try to load locally saved dates.
        2. Try to load the dates from S3, then save locally. 
        3. Try to query NAP for the dates.
        If refresh=True,
        3. Query NAP, save a local copy of dates, and if upload=True, save a s3 copy of dates.
        
        Add SQL parameters with kwargs:
        For dates, date coverage extends from earliest date to <date> + <horizon> years.

        Config() Attribute Parameters
        -----------------------------
        basepath, vars, LOG, conn

        Parameters
        ----------
        file : str
            file is the root name (no extensions) for both the SQL file and for the locally/s3-saved CSV file.
        s3_key : str
            S3 key where to download or upload the data.
        conn : str
            Name of database connection string
        refresh : bool
            Query the database for the reference data?
        upload : bool
            If refresh=True, upload queried data to S3?
            Set to True for regular weekly runs.
            Set to False for adhoc work and backcasting.
        sql_params : dict
            SQL format strings as {key:value}, where the SQL might be "select * from mytable where dept={key};"
            Ex. for reference='dates':
                date : str
                    Date as YYYY-MM-DD. This function will retrieve dates through that fiscal year + horizon.
                future_yrs : int
                    How many fiscal years into the future to return.

        Returns
        -------
        pandas.DataFrame
            Data corresponding to 

        if refresh=TRUE, upload=TRUE, send results to
        s3://{self.s3}/{s3_key}/{file}.csv

        Examples
        --------
        dates = _get_reference_data(file='dates', s3_key='input/lookup/', refresh=False)              # get dates from local or S3
        dates = _get_reference_data(file='dates', s3_key='input/lookup/', refresh=True)               # get dates from database
        dates = _get_reference_data(file='dates', s3_key='input/lookup/', refresh=True, upload=True)  # get dates from database and save a copy in S3
        """ 
        LOCAL_FILE = f"{self.basepath}/data/{file}.csv"
        S3_BUCKET = self.vars['s3']
        S3_FILE = f"{s3_key}/{file}.csv"
        DB = conn or self.conn['nap']  # Database to query, defaulting to NAP
        
        if not refresh:
            try: # to load local file first
                df_ref = pd.read_csv(filepath_or_buffer=LOCAL_FILE)
                self.LOG.info(f"Loading {LOCAL_FILE}")
            except FileNotFoundError: # if not found, then load from s3
                try:
                    df_ref = nordypy.s3_to_pandas(bucket=self.s3, s3_filepath=S3_FILE).reset_index()
                    self.LOG.info(f"Loading from s3://{self.s3}/{S3_FILE}")
                    df_ref.to_csv(LOCAL_FILE, index=False)
                except ClientError as ex:
                    if ex.response['Error']['Code'] == 'NoSuchKey':
                        refresh = True
                    else:
                        self.LOG.error(ex)
            
        if refresh:
            self.LOG.info(f"Querying {file} from {DB}")
            # First, check for DB creds in ENV
            if DB not in os.environ:
                self._nordypy_creds(connections=[DB], refresh=True)

            # Then query data from DB
            sql = nordypy.read_sql_file(f"{self.basepath}/src/queries/ref/{file}.sql").format(**sql_params)

            df_ref = nordypy.database_get_data(database_key=DB, sql=sql, as_pandas=True)
            # Note: if query fails, ENV creds may need to be refreshed.

            # Save locally and re-read data for consistency between queries and CSV
            # re-read the csv because dates are datetime.date when freshly queried but object strings when loaded from saved csv
            df_ref.to_csv(LOCAL_FILE, index=False)
            df_ref = pd.read_csv(filepath_or_buffer=LOCAL_FILE)
            df_ref = self._string_to_datetime(df=df_ref)

            # Save to S3
            if upload:
                nordypy.pandas_to_s3(data=df_ref,
                                    bucket=S3_BUCKET,
                                    s3_filepath=S3_FILE,
                                    header=True,
                                    delimiter=',')


        # Note: "UnboundLocalError: local variable 'df_ref' referenced before assignment"
        # likely means ExpiredToken. Try restarting aws-okta
        return self._string_to_datetime(df=df_ref)


    def _date_to_week_lookup(self, dates, date_lkp, date_col='week'):
        """ Find the fiscal week corresponding to the date or vice versa. 
        This fiscal week is used for filtering data and determining forecast window start.
        Note: this function can also be used with fiscal day, month, and year (see examples below).

        Attribute Parameters
        --------------------
        events

        Parameters
        ----------
        dates : pandas.DataFrame
            DataFrame with columns WK_IDNT | WK_START_DT | DAY_DT
        date_lkp : int, str, or datetime.date
            Look up the corresponding WK_IDNT or WK_START_DT corresponding to this variable.
            If WK_IDNT is entered, return WK_START_DT and vice-versa.
        date_col : str
            Name of lookup column corresponding to date.
            This string will have '_num' or '_start_day' appended to it.
            ex. date_col='week', then use columns 'week_num' and 'week_start_day'

        Returns
        -------
        int or datetime.date
            Starting week of forecast as YYYYFW or datetime.date(YYYY, MM, DD), ex. 202001 or datetime.date(2020, 02, 02)
            Convert to YYYY-MM-DD string with datetime.date(2020, 02, 02).strftime(format="%Y-%m-%d")

        Examples
        --------
        _date_to_week_lookup(dates=df, date_lkp=202030, date_col='week')                      # datetime.date(2020, 8, 23)
        _date_to_week_lookup(dates=df, date_lkp='202030', date_col='week')                    # datetime.date(2020, 8, 23)
        _date_to_week_lookup(dates=df, date_lkp=202007, date_col='month')                     # datetime.date(2020, 8, 2)
        _date_to_week_lookup(dates=df, date_lkp=datetime.date(2020, 8, 23), date_col='week')  # 202030
        _date_to_week_lookup(dates=df, date_lkp='2020-08-23', date_col='week')                # 202030
        _date_to_week_lookup(dates=df, date_lkp='2020-08-23', date_col='day')                 # 2020204
        _date_to_week_lookup(dates=df, date_lkp='2020-08-23', date_col='month')               # 202007
        _date_to_week_lookup(dates=df, date_lkp='2020-08-23', date_col='year')                # 2020
        """
        date_num = f"{date_col}_num"
        dates1 = self._string_to_datetime(df=dates.sort_values(by=['day_date']))

        # From integer date to start date
        try:
            other_dt = dates1.loc[dates1[date_num] == int(date_lkp), 'day_date'].min()
            return other_dt.to_pydatetime().date()  # <-- datetime.date. For string, add .isoformat()

        except (TypeError, ValueError):
            if isinstance(date_lkp, datetime.date):
                date_lkp = date_lkp.isoformat()
            
            # check if date string is valid
            try:
                datetime.date.fromisoformat(date_lkp)
            except ValueError:
                self.LOG.exception(f'{date_lkp} is not a valid date string.')
                raise
            if isinstance(datetime.date.fromisoformat(date_lkp), datetime.date):
                # From date to integer date
                return dates1.loc[dates1['day_date'] == date_lkp, date_num].min()

    def _dates_around_date(self, index, start, horizon=-1):
        """ Return dates around a specific date. 
        
        Parameters
        --------------------
        index: pandas.core.series.Series
            Date index. The function will de-duplicate and sort the index.
        start: int
            index element to match, e.g. 202037 if fiscal week
        horizon: int
            number of weeks before or after a specific week, negative if before the week

        Return
        ----------
        selected_list: list
            a list of dates around the specified date

        Example
        ----------
        _dates_around_date(index=dates['WK_IDNT'], start=202040, horizon = -3)
        > [202037, 202038, 202039]
        _dates_around_date(index=dates['WK_IDNT'], start=202040, horizon = 1)
        > [202041]
        """
        index1 = index.drop_duplicates().sort_values().tolist()
        
        try:
            idx = index1.index(start)
            if horizon > 0:
                selected_list = index1[idx + 1: idx + horizon + 1]

            else:
                selected_list = index1[idx + horizon: idx]

            return selected_list

        except ValueError:
            self.LOG.error(f"The date {start} doesn't exist in the list")


    ###########################
    ##   UTILITY FUNCTIONS   ##
    ###########################
    # General standalone functions (i.e. static methods)

    @staticmethod
    def _blank_str_check(string, delim_before='', delim_after=''):
        """ Check if string is empty and format it to blank string.
        If string is not empty, add a delimiter before/after if needed.
        This function is used when a user calls a function
        - that doesn't require any string adjustments without an optional parameter, or
        - requires string adjustments with an optional parameter.
        Ex. delete_from_s3(bucket='mybucket', key='mykey', [mypartition='2020']) could delete either
        s3://mybucket/mykey/                without optional parameter partition, or
        s3://mybucket/mykey/mypartion=2020  with optional parameter partition='2020'.
        but then an if/else statement would be needed to add the '=' between mypartition and the partition id when that arg is called.

        This function is the if/else statement but flexible for multiple applications, like filenames.

        Parameters
        ----------
        string : string
            Variable to check if blank.
        delim_before, delim_after : string
            Delimeter to add before/after the string if the string is not blank.

        Returns
        -------
        string
            Blank string if string is empty or None, and string with delimiters if string exists.
        """
        if not string or not string.strip():
            label = ''
        else:
            label = delim_before + string.strip() + delim_after
        return label
    
    @staticmethod
    def _nearest(items, marker, comparison='<'):
        """ Find the element nearest to the marker. Useful for finding most recent file.

        Parameters
        ----------
        items : array
            Array of items on which we can perform addition, subtraction, abs().
        marker : int, numeric, datetime, etc...
            Item nearest to this marker value.
        comparison : str
            logical comparison for items relative to marker.
            ex. less than (<) = only items that occur before the marker.
            If None, then search both directions for nearest item.

        Returns
        -------
        item
            Item nearest to the marker value. Ties go to the minimum value.

        Example
        -------
        nearest(items=[1,4,5,7,10], marker=5, comparison='<')
        >>> 4
        nearest(items=[10,7,5,4,1], marker=5, comparison='<') # items order does not matter
        >>> 4
        nearest(items=[1,4,5,7,10], marker=5, comparison='<=')
        >>> 5
        nearest(items=[1,4,5,7,10], marker=5, comparison='==')
        >>> 5
        nearest(items=[1,4,5,7,10], marker=5, comparison='>')
        >>> 7
        # tie goes to lower value
        nearest(items=[1,4,5,7,10], marker=6, comparison=None)
        >>> 5
        """
        # string to operator lookup
        lkp = {'<':  operator.lt,
            '<=': operator.le,
            '=':  operator.eq,
            '==': operator.eq,
            '>=': operator.ge,
            '>':  operator.gt
            }

        if comparison:
            fn = lkp[comparison] # select comparison logic from lookup
            items = [i for i in items if fn(i,marker)] # filter items by comparison logic

        # Return nearest match
        return min(items, key=lambda x: abs(x - marker)) 

    @staticmethod
    def _string_to_datetime(df, columns=None, format="%Y-%m-%d", date=False):
        """ Convert date columns from string to datetime format.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe with date columns in string format
        columns : list(str)
            Columns to convert.
        format : str
            Format of date string from pandas.to_datetime()
        date : bool
            If True, convert date back to date object. Required for Parquet formats.

        Returns
        -------
        pandas.DataFrame
            Same as input data, except with datetime64[ns] (or object if date=True) columns.
        """
        df1 = df.copy()

        # Common date formats to convert if columns is None
        if not columns:
            c0 = ['WK_START_DATE', 'WEEK_START_DATE', 'WK_END_DATE', 'WEEK_END_DATE',
                  'DS', 'DAY', 'DATE', 'DT', 'DAY_DATE', 'START_DATE', 'END_DATE',
                  'RUN_DATE', 'FCST_DATE', 'FORECAST_DATE', 'LAST_YEAR_DAY_DATE', 'LAST_YEAR_DAY_DATE_REALIGNED']
            c1 = [c.replace("_DATE", "_DAY") for c in c0]
            c2 = [c.replace("_DATE", "_DT") for c in c0]
            columns = list(set(c0 + c1 + c2)) 
            columns = columns + [c.lower() for c in columns]
            columns = list(df.columns.intersection(columns))

        df1[columns] = (df1[columns].apply(pd.to_datetime, format=format))
        
        # convert datetime[ns] to date object
        if date:
            for d in columns:
                df1[d] = df1[d].dt.date
        return df1

    @staticmethod
    def _cartesian_product_multi(*dfs):
        """ Cartesian product https://stackoverflow.com/a/53699013/4718512

        Parameters
        ----------
        *dfs : pandas.Dataframe(s)

        Examples
        --------
        cartesian_product_multi(
            pd.DataFrame({'king':['A','B']}),
            pd.DataFrame({'kong':[1,2,3]}),
            pd.DataFrame({'smash':[9,8]})
        )
        >  	king 	kong 	smash
        0 	A 	    1 	    9
        1 	A 	    1 	    8
        2 	A 	    2 	    9
        3 	A 	    2 	    8
        4 	A 	    3 	    9
        5 	A 	    3 	    8
        6 	B 	    1 	    9
        7 	B 	    1 	    8
        8 	B 	    2 	    9
        9 	B 	    2 	    8
        10 	B 	    3 	    9
        11 	B 	    3 	    8
        """
        def cartesian_product(*arrays):
            la = len(arrays)
            dtype = np.result_type(*arrays)
            arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
            for i, a in enumerate(np.ix_(*arrays)):
                arr[..., i] = a
            return arr.reshape(-1, la)

        idx = cartesian_product(*[np.ogrid[:len(df)] for df in dfs])
        df_out = pd.DataFrame(
            np.column_stack([df.values[idx[:, i]] for i, df in enumerate(dfs)]))
        cols = []
        for d in dfs:
            cols = cols + list(d.columns)
        df_out.columns = cols
        return df_out

    @staticmethod
    def _uncertainty(uncertainty, value=None):
        """ Add uncertainty (u) in quadrature.

        For addition (or subtraction), where y = a + b:
        u_y = sqrt(u_a^2 + u_b^2) 

        For multiplication, where y = a * b:
        u_y/y = sqrt((u_a/a)^2 + (u_b/b)^2)

        Note: This function cannot handle division because the y value is not straightforward.
        ex. from below:
        4 / 5 / 6 = 0.133
        5 * 6 / 4 = 7.5
        4 / 5 * 6 = 4.8
        and all these values would multiply sqrt((1/4)^2 + (2/5)^2 + (3/6)^2) to get final uncertainty of each calculation.

        Parameters
        ----------
        uncertainty : array
            Uncertainty
        value : array
            Values. Used for multiplication.

        Returns
        -------
        numeric

        Examples
        --------
        df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
        uncertainty(uncertainty=df['a'])  # 3.74 for addition, i.e. 4±1 + 5±2 + 6±3 = 15±3.74
        uncertainty(uncertainty=df['a'], value=df['b'])  # 82.49 for multiplication, i.e. 4±1 * 5±2 * 6±3 = 120±82.49
        """
        import numpy as np
        # multiplication or division requires uncertainties to be scaled to original value
        # addition or subtraction requires no modification to uncertainties
        # so if addition or substraction, then divide by 1 (no scaling)
        if value is None:
            value = 1

        # Note: for multiplication or division, component uncertainties need to 
        # be multipled by final calculated value to derive uncertainty.
        # The np.prod() below only works with product (i.e. multiplcation), not with any kind of division.
        return np.sqrt(np.sum(np.square(uncertainty/value))) * np.prod(value)

    @staticmethod
    def _convert_to_list(x):
        """
        Convert any type to list. For use with schedule yaml. 
        
        Args:
            x (list, str, int, or float): input
        Returns:
            list: input as a list
        """
        # already a list
        if type(x)==list:
            pass
        # tuple to list
        if type(x)==tuple:
            x = list(x)
        elif type(x)==str: 
            # string representation of list to list
            if '[' in x: 
                x = x.strip('][').split(', ')
            # string representation of tuple to list
            elif '(' in x: 
                x = x.strip(')(').split(', ')
            # string with commas
            elif ',' in x: 
                x = x.split(', ')
                # list of ints 
                try:
                    x = [int(i) for i in x]
                except: 
                    pass
                # list of floats 
                try:
                    x = [float(i) for i in x]
                except: 
                    pass
            # string to list with one item
            else: 
                x = [x]
        # int to list with one item
        elif type(x)==int: 
            x = [x]
        # float to list with one item
        elif type(x)==float: 
            x = [x]
        else: 
            pass
        return x

class util(Config):
    """ Utility functions that don't quite fit in DSConfig. 

    Attributes
    ----------
    variables : str
        Name of key in YAML file holding variables like s3 keys and table names.
    envi : str
        Name of key in YAML file holding environment variables like prod/nonprod
        connnection strings. ex. 'prod'
    """

    def __init__(self, variables='universal', envi=None):
        super().__init__(variables, envi)

    @staticmethod
    def filter_kwargs(fn=None, *, exclude=['self']):
        """  This function filters kwargs to avoid:
        TypeError: function() got an unexpected keyword argument 'arg'

        It is written as a decorator so extra args can be filtered automatically,
        for example, when a dict of many parameters is passed to a function that only uses a few of them.
        Decorator with optional args: https://stackoverflow.com/a/24617244/4718512

        
        Examples
        --------
        def demo(a,b):
            # This function requires args a and b.
            print(f"{a} {b}")
    
        demo(a='foo',b='bar')
        > foo bar

        # Extra kwargs will cause function error
        mykwargs = {'a':'cat', 'b':'dog', 'c':'bigfoot'}
        demo(**mykwargs)  # error because extra arg c is passed to function
        > TypeError: demo() got an unexpected keyword argument 'c'

        # Redefine the function, allowing extra kwargs
        @util.filter_kwargs
        def demo1(a,b):
            print(f"{a} {b}")
        
        demo1(**mykwargs) # Now the function works with extra args
        > cat dog

        @util.filter_kwargs(exclude=['c'])  # exclude arg c from passed kwargs
        def demo2(a,b,c=1):
            print(f"{a} {b} {c}")

        demo2(**mykwargs)
        > cat dog 1

        """
        def _decorate(function):
            # When decorator is called with optional args, decorator is called with function=None, and a decorating function is returned
            @functools.wraps(function)
            def wrapped_function(*args, **kwargs):
                # Get function args
                fn_args = list(function.__code__.co_varnames)
                # Filter out escluded args (like self in class methods)
                fn_args = [i for i in fn_args if i not in exclude]
                # Filter kwargs with desired args
                new_kwargs = {k:kwargs[k] for k in set(kwargs.keys()) & set(fn_args)}
                # Return function with only desired kwargs
                return function(*args, **new_kwargs)
            return wrapped_function

        if fn:
            # When decorator is called with no optional args, then treat it like:
            # @decorator
            # def my_function ...
            return _decorate(fn)

        return _decorate
