import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import yaml
import logging
import subprocess
import psutil
from timeit import default_timer as timer
import datetime

class Config(object):
    """Config allows users to easily set repo-wide variables, including:
    1. self.vars = universal variables like s3 buckets and keys
    2. self.conn = MLP connection strings based on environment (local, prod, nonprod).
    3. self.args = parsed arguments
    4. self.ENV = ENV variables like worker IDs for parallelized Airflow jobs.
    These variables are specified in config/*.YAML files.

    Config also creates a standard logger (self.LOG) named after the child class.
    
    How to use it:
    class MySubclass(Config):
        def __init__(self, variables='universal', envi=None):
            super().__init__(variables, envi)
            # Pass variables and envi from subclass to Config superclass
            # Rest of your code here.

    Attributes
    ----------
    variables : str
        Name of key in config/config.YAML file holding variables like s3 keys and table names.
    envi : str
        Environment, like 'nonprod', 'prod', or your LANID.
        This is used to set self.ENVIRONMENT (for connection keys) and 
        self.ENVI_TYPE (for noting if we're running in prod or nonprod. Useful for setting LDAP='auth' for teradata connections)

    Returns self.* Attributes
    -------------------------
    LOG : logging.Logger
        Logger. Add statements from the logger module like self.LOG.info('job finished')
    ENVIRONMENT : str
        User's LANID, prod, or nonprod corresponding to the environment for connection look-ups.
    ENVI_TYPE: str
        'nonprod' or 'prod'. Nonprod includes local LANID and Kubernetes nonprod. 
        Used for variables that differ between prod and nonprod, like databases.
    vars : dict
        Variables found under config_yaml['variables']
    ENV : dict
        Specific ENV variables assigned as values to keys.
        Data types of the input strings are inferred.
    conn : dict
        Connection strings found under config_yaml['connections'][envi]
    args : argparse.Namespace
        Command-line arguments specified in parsed_args.yaml, accessed with
        self.args.foo, self.args.bar, etc.
    """

    def __init__(self, variables='universal', envi=None):
        super(Config, self).__init__()
        # Set up logging.
        logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        self.LOG = logging.getLogger(type(self).__name__)  # name the logger after the class that loads it.
        self.LOG.setLevel(logging.DEBUG)

        # Root folder
        # Note: each dirname corresponds to the parent directory of the current file location.
        self.basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Use ENV variables to detect the environment in which the code is run
        # ENVIRONMENT is used for detecting connections (prod, nonprod, lanid)
        # ENVI_TYPE is prod or nonprod for switching endpoints or databases
        envi = envi or os.getenv('LANID') or os.getenv('RUNNING_ENV', 'nonprod')
        self.ENVIRONMENT = envi.lower().replace('production', 'prod')
        self.ENVI_TYPE = (self.ENVIRONMENT=='prod')*'prod' or (self.ENVIRONMENT!='prod')*'nonprod'

        # Create parsed arguments. 
        # Run this early so users can pass parsed args to other initialization functions.
        self.args = self._make_args()
        self.LOG.info(self.args)

        # Load the configurations for variables and environment
        self.vars = self._get_config(key=self.args.variables or variables)

        # Read ENV variables
        self.ENV = self._env_to_kwarg()

        # Set-up connections.
        self.conn = self._detect_connections()

        # Record git commit hash
        self.git = self._git_describe()

    ############################
    ##   YAML CONFIGURATION   ##
    ############################

    def _get_config(self, file='config/config.yaml', key=None, all_keys=False):
        """ Load configuration from yaml. The configuration holds variables used across scripts.
        Different sets of variables can be stored in this file, ex. prod vs nonprod environments,
        universal variables, test variables, table names, tags, etc.
        This method uses dict.get() to allow it to return None, which can be
        useful for coalescing variables, e.g. variable = automatic or user_defined or default

        Parameters
        ----------
        file : str
            File_path/file_name of the configuration file.
        key : str
            Name of first key in the configuration file to access.
        all_keys : bool
            Retrieve the entire config file instead of a specific key?

        Returns
        -------
        dict
            Dictionary of key-value pairs located in the config.yaml.

        Examples
        --------
        # Access the prod/nonprod parameters and universal variables as envi_config and config, respectively.
        envi_config = Config()._get_config(all_keys=True)  # for all variables
        config = Config()._get_config(key='universal')  # for constants
        config['s3']  # returns s3 bucket
        """
        with open(f"{self.basepath}/{file}", 'r') as cf_file:
            try:
                config = yaml.safe_load(cf_file)
            except yaml.YAMLError as exc:
                print(exc)

        if all_keys:
            return config
        else:
            return config.get(key)
    
    #############################
    ##  ENVIRONMENT VARIABLES  ##
    #############################

    def _detect_connections(self):
        """ Use self.ENVIRONMENT to set the appropriate credentials.

        The init() logic checks environment in this order:
        1. if ENV LANID --> nonprod > lanid credentials
        2. if RUNNING_ENV --> prod > k8s credentials
        3. else --> nonprod > k8s credentials

        Attribute Parameters
        --------------------
        ENVIRONMENT

        Returns
        -------
        self.envi : dict
            Dictionary of connection-type : connection namekey-value pairs
            located in the config.yaml.
        """
        connections = self._get_config(file='config/connections.yaml', key='connections')  # Load all connections
        return connections[self.ENVIRONMENT]  # subselect connections for current ENVIRONMENT

    def _string_parser(self, string):
        """ ENV variables are stored as strings. This function parses those strings into
        numbers, booleans, or leaves them as strings.
        Note: Booleans are not case-sensitive, so 'true', 'True', 'TRUE', 'tRuE' all evaluate to True.

        Parameters
        ----------
        string : str
            string to parse

        Returns
        -------
        int, float, bool, str, or None

        Examples
        --------
        [(i,type(self._string_parser(i))) for i in [None, '4', '4.8', 'True', 'gesundheit']]
        > [(None, NoneType),
           ('4', int),
           ('4.8', float),
           ('True', bool),
           ('gesundheit', str)]
        """
        if string is None:
            return None

        # recursively parse items in list
        if isinstance(string, list):
            return [self._string_parser(s) for s in string]

        try:
            val = int(string)
        except ValueError:
            try:
                val = float(string)
            except ValueError:
                bool_dict = {'TRUE': True, 'FALSE': False}
                val = bool_dict.get(string.upper(), string)
        return val

    def _listify(self, item):
        """ Check if item is a list, and if not, make it a list.
        Useful for ensuring variables that are intended to be iterables are truly
        iterable, as with for loops.
        ex. 
        ARG = 'single_item'
        for i in ARG:
            print(i)  # prints 's','i','n','g','l'... instead of 'single_item'

        Parameters
        ----------
        item : any
            Item to check

        Examples
        --------
        for i in [1,'foo', ['foo'], ['foo','bar'], 'bar', [1,2],[608]]:
            print(_listify(i))
        #[1]
        #['foo']
        #['foo']
        #['foo', 'bar']
        #['bar']
        #[1, 2]
        #[608]
        """
        if item:
            if isinstance(item, list):
                return item
            else:
                return [item]

    def _env_to_kwarg(self):
        """ Collect ENV variables in a predefined dict where
        key = the variable name used in this class
        value = the ENV variable name.

        Attribute Parameters
        --------------------
        _get_config(), _string_parser()

        Returns
        -------
        env_kwargs : dict
            Argument names and their values.
        """

        # Load placeholder ENV variable dict
        placeholder = self._get_config(file='config/env_variables.yaml', all_keys=True)

        # Parse ENV variables matching placeholder.values().
        env_kwargs = {k: self._string_parser(os.environ.get(v, None))
                      for k, v in placeholder.items()}
        return env_kwargs

    def _git_describe(self, arg="--always") -> str:
        """ Return output of `git describe`, i.e. LatestTag_#CommitsSinceTag_CommitHash.
        This is useful for tracking codebase.
        from: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

        ex. v1.0.4-14-g2414721
        would be:
        TAG: v1.0.4
        Commits since that tag: 14
        commit hash: 2414721 (The -g is a prefix)

        Note: Version can have hyphens, so when string-splitting to get commit, start from the end of the string.
        Get the commit hash with _git_describe().split('-')[-1].replace('g','',1)

        Parameters
        ----------
        arg : str
            See git describe --help

        Returns
        -------
        str
            Output of `git describe`, i.e. LatestTag_#CommitsSinceTag_CommitHash

        Examples
        --------
        _git_describe()
        > 'v1.0.4-14-g2414721'
        """
        return (subprocess.check_output(["git", "describe", arg], cwd=self.basepath).
                decode('ascii').strip())

    ##########################
    ##   PARSED ARGUMENTS   ##
    ##########################

    def _make_argparser(self):
        """ Create command-line parsed arguments from a YAML file.
        Top-level YAML keys do not matter but must be unique. Each of these 
        top-level keys represent a separate call to argparse.add_argument() in the
        for-loop of this function.

        Attribute Parameters
        --------------------
        _get_config()

        Returns
        -------
        argparse.Namespace
            Args stored inside the class.
        """
        parser = argparse.ArgumentParser(allow_abbrev=False) 
        # allow_abbrev=False solves ikernel ambiguous option -f error since we have arguments that begin with f.
        arg_dict = self._get_config(file='config/parsed_args.yaml', all_keys=True)  # dict of args

        for v in arg_dict.values():
            addtl_args = v.copy()  # preserve original config
            names = addtl_args.pop('name')  # Pass argument name(s) as *args
            # yaml reads values as string and does not preserve type, so set it here, defaulting to string.
            # If action="store_true", then type must not be set.
            if addtl_args.get('action', None)=='store_true':
                addtl_args.pop('type', None)
            else:
                addtl_args['type'] = eval(addtl_args.get('type', 'str'))
            parser.add_argument(*names, **addtl_args)
        
        return parser

    def _make_args(self):
        """ Create command-line parsed arguments from a YAML file.
        Top-level YAML keys do not matter but must be unique. Each of these 
        top-level keys represent a separate call to argparse.add_argument() in the
        for-loop of this function.

        Attribute Parameters
        --------------------
        _get_config(), self.argparser

        Returns
        -------
        argparse.Namespace
            Args stored inside the class.
        """
        parser = self._make_argparser()
        args, unknowns_list = parser.parse_known_args()
        return args

    # def _make_args(self):
    #     """ Create command-line parsed arguments from a YAML file.
    #     Top-level YAML keys do not matter but must be unique. Each of these 
    #     top-level keys represent a separate call to argparse.add_argument() in the
    #     for-loop of this function.

    #     Attribute Parameters
    #     --------------------
    #     _get_config(), self.argparser

    #     Returns
    #     -------
    #     argparse.Namespace
    #         Args stored inside the class.
    #     """
    #     parser = argparse.ArgumentParser(allow_abbrev=False) 
    #     # allow_abbrev=False solves ikernel ambiguous option -f error since we have arguments that begin with f.
    #     arg_dict = self._get_config(file='config/parsed_args.yaml', all_keys=True)  # dict of args

    #     for v in arg_dict.values():
    #         addtl_args = v.copy()  # preserve original config
    #         names = addtl_args.pop('name')  # Pass argument name(s) as *args
    #         # yaml reads values as string and does not preserve type, so set it here, defaulting to string.
    #         # If action="store_true", then type must not be set.
    #         if addtl_args.get('action', None)=='store_true':
    #             addtl_args.pop('type', None)
    #         else:
    #             addtl_args['type'] = eval(addtl_args.get('type', 'str'))
    #         parser.add_argument(*names, **addtl_args)

    #     args, unknowns_list = parser.parse_known_args()
    #     return args

    ###################
    ##   UTILITIES   ##
    ###################
    def _memory_check(self, text=""):
        """ Check memory usage.
        Parameters
        ----------
        text : str
            String to attach to debug statement, ex. f"Dept {d}."

        Returns
        -------
        log statements

        Examples
        --------
        self.__memory_check()
        > Memory use= 0.1 Gb (1% of 16 Gb) for process PID 21535
        """
        # Create OS process (or Linux thread ID)
        try:
            pp = psutil.Process(pid=None) 
        except (NoSuchProcess, AccessDenied) as e:
            self.LOG.exception(e)
            return
        with pp.oneshot():
            GB = pp.memory_info()[0]/2.**30  # memory use in GB
            pct = pp.memory_percent()
        total_memory = GB/(pct/100)
        self.LOG.debug(f"{text} Memory use: {round(GB,1)} Gb ({round(pct)}% of {round(total_memory)} Gb) for process PID {pp.pid}")

    def _time(start: datetime.datetime = None):
        """ Timer in UTC time.
        If no function args, log the start time in UTC.
        If start is provided, return current time and minutes elapsed since start.

        Returns
        -------
        datetime.datetime -or- tuple(datetime.datetime, float)

        Examples
        --------
        start = _time()
        time.sleep(3)
        end_and_elapsedMinutes = _time(start=start)
        print(end_and_elapsedMinutes) # tuple
        > (datetime.datetime(2022, 12, 21, 6, 32, 6, 862238, tzinfo=datetime.timezone.utc),
        0.05007873333333333)
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        if isinstance(start, datetime.datetime):
            return now, (now - start).total_seconds() / 60
        else:
            return now


    def _memory_and_cpu():
        """ Return the RAM (in Gb) and CPU count of the current environment for metadata."""
        ram = psutil.virtual_memory().total / 2.**30  # RAM, Gb
        cpus = psutil.cpu_count()  # includes threads
        return ram, cpus
