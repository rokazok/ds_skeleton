# ds_skeleton
Data Science skeleton is a set of parent classes to facilitate and standardize data science projects, specifically with data preparation and data import/export. I wrote these classes when I led teams on forecasting initiatives where developers would create different models. 

The repo structure includes in `src/helers`:
1. Config from [initializer.py](src/helpers/initializer.py): the "skeleton," with basic functions that can be used in any data science project and without any online functions.
2. DSConfig from [datascience_initializer.py](src/helpers/datascience_initializer.py): the "muscles," with database connection methods, reference data refreshes, data import/export methods for AWS S3. These were specific to my job, leveraging services for credentials, and therefore will not work without refactoring. Inherits Config.
3. ModelParent from [training_parent_class.py](src/helpers/training_parent_class.py): the "organs," with methods custom to the application (here, forecasting). Users would need to refactor this to support their specific use-case.
4. (Bonus) [parallelizer.py](src/helpers/parallelizer.py): has 2 classes for parallelizing jobs. The first, AirflowParallelizer, just facilitates creating DAGs for the internal machine learning platform and has limited external use. The second, `LocalParallelize`, creates parallel jobs across cores on a single machine. Its `_parallel_over_list()` method is useful for flexibly passing in different combinations of functions and arguments.


Depending on use case, an example of how to use these classes is creating a new class that inherits the core methods of DSConfig:
```python
from src.helpers.datascience_initializer import DSConfig

class MyNewClass(DSConfig):
    """ This class has functions standard to models, including data pulls, output formatting, and uploading.

    Attributes
    ----------
    variables : str
        Name of key in YAML file holding variables like s3 keys and table names.
    envi : str
        Name of key in YAML file holding environment variables like prod/nonprod
        connnection strings. ex. 'prod'
    parallel_name : str
        Name of CSV file in S3 with parallel worker assignments.
    fcst_dt : int
        Date of forecast as YYYY-MM-DD, ex. 2020-12-31.
    model, model_version : str
        Name of model and version
    """
    
    def __init__(self, variables='universal', envi=None, parallel_name=None, fcst_dt=None, model=None, model_version=None, db_creds=True, ref_data=True):
        super(MyNewClass, self).__init__(variables=variables, envi=envi, parallel_name=parallel_name, fcst_dt=fcst_dt, db_creds=db_creds, ref_data=ref_data)

        # Note: DSConfig() sets self.fcst_dt
        self.initial_fcst_dt = fcst_dt or self.args.fcst_dt   # record initial input date
        self.model = model or self.args.model
        self.model_version = model_version
        # more initial variables here

    # add new methods here
```
