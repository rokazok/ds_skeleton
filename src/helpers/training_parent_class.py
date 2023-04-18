import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import re
import boto3
from src.helpers.datascience_initializer import DSConfig
from timeit import default_timer as timer
import nordypy
from datetime import datetime





class ModelParent(DSConfig):
    """ This class has functions standard to models, including data pulls, output 
    formatting, and uploading.

    How to use it with your custom MyModelClass:
    class MyModelClass(ModelParent):
        def __init__(self, variables='universal', envi=None, fcst_dt=None, parallel_name=None, model='myModel', model_version='1'):
            super(MyModelClass, self).__init__(variables=variables, envi=envi, fcst_dt=fcst_dt, parallel_name=parallel_name, model=model, model_version=model_version)
            # Rest of your code here.

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
        super(ModelParent, self).__init__(variables=variables, envi=envi, parallel_name=parallel_name, fcst_dt=fcst_dt, db_creds=db_creds, ref_data=ref_data)

        # Note: DSConfig() sets self.fcst_dt
        self.initial_fcst_dt = fcst_dt or self.args.fcst_dt   # record initial input date
        self.model = model or self.args.model
        self.model_version = model_version
        self.channels = self._listify(self.args.channels) or ['110', '120', '210', '250'] # Channels to model. These should match DSConfig._parse_channel() outputs.
        self.runtime_log_lst=[]
        if self.args.departments:
            self.departments = self._listify(self._string_parser(self.args.departments))
        elif db_creds:  # if user is connected to AWS and databases
            # As of 03/2022, most jobs are parallelized across groups of departments in a for-loop.
            # The code below determines which departments are assigned to the current worker.
            self.parallel_name = parallel_name or self.args.parallel_name or 'weekly_prod_full' # name of assignments file in s3
            self.assignments = self._get_worker_assignments(parallel_name=self.parallel_name) # get assignments dataframe from s3
            # Passing assignments to itself will return only the grouping variables instead of inner-joining and subsetting the training data.
            # Since model training is currently performed at the department level (and input data is already a subset of all data), 
            # the single grouping variable 'dept_num' is converted to a list.
            try:
                worker_depts_from_s3 = self._get_worker_data(df_full=self.assignments, assignments=self.assignments)['dept_num'].to_list()
            except TypeError:
                worker_depts_from_s3 = None
            self.departments = self._listify(worker_depts_from_s3)

            # Data dictionary for training data in s3
            self.actuals_dtypes = {
                #'channel_num': 'int',
                'epm_choice_num': 'int',
                'class_num': 'int',
                'sbclass_num': 'Int64',
                'brand_name': 'string',
                'prmy_supp_num': 'string',
                'manufactuer_num': 'string',
                'color_num': 'string',
                'color_desc': 'string',
                'epm_style_num': 'string',
                'web_style_num': 'float64',
                'style_desc': 'string',
                'live_date': 'string',
                'week_num': 'Int32',
                'demand': 'float64',
                'demand_store_take': 'float64',
                'demand_owned': 'float64',
                'demand_dropship': 'float64',
                'boh': 'Int64',
                'boh_sku_count': 'Int64',
                'boh_store': 'Int64',
                'boh_store_ct': 'Int64',
                'boh_store_sku_ct': 'Int64',
                'dropship_boh': 'float64',
                'dropship_sku_ct': 'float64',
                'regular_price_amt': 'float64',
                'current_price_amt': 'float64',
                'current_price_type': 'string',
                'current_price_event': 'string',
                'event_tags': 'string',
                'allocated_qty': 'Int64',
                'received_qty': 'float64',
                'rp_ind': 'Int8',
                'averagerating': 'float64',
                'product_views': 'float64',
                'add_to_bag': 'float64',
                'channel_brand': 'string',
                'selling_channel': 'string',
                'regular_event_name': 'string',
                'loyalty_event': 'string',
                'dept_num': 'string'}

            # Athena/Glue data types: https://docs.aws.amazon.com/athena/latest/ug/data-types.html
            # LAZY TO AVOID int type errors with string floats like '21.0' and Nones
            self.forecast_parquet_dtypes = {'dept': 'varchar', 'class_num': 'varchar', 'supplier': 'varchar', 'color_num': 'char(3)', 'cc_idnt': 'varchar',
                            'y': 'float', 'y_high': 'float', 'y_low': 'float', 'date': 'date', 'run_date': 'date', 'fcst_dt': 'date',
                            'model': 'varchar', 'channel_brand': 'char(14)', 'selling_channel': 'char(6)', 'channel_country': 'char(2)'}
            self.metadata_parquet_dtypes = {'dept': 'varchar', 'run_date': 'date', 'fcst_dt': 'date',
                            'model': 'varchar', 'channel_brand': 'char(14)', 'selling_channel': 'char(6)', 'channel_country': 'char(2)',
                            'start_run_time': 'timestamp', 'end_run_time': 'timestamp', 'environment': 'varchar',
                            'git': 'varchar', 'memory': 'float', 'CPUs': 'float'}

    # BELOW = General helper functions
    @staticmethod
    def _typecheck(obj, t: type):
        """ Check if an object is the correct type. If not, throw an error.
        This is a one-line wrapper around isinstance() to check function variables are correctly entered.
        
        Parameters
        ----------
        obj : variable to check
        t : type, i.e. int, str, list, dict, np.ndarray, pd.DataFrame

        Examples
        --------
        typecheck({'a':1}, int)           # TypeError: Arg type is 'dict' and it should be 'int'
        typecheck({'a':1}, dict)
        typecheck({'a':1}, pd.DataFrame)  # TypeError: Arg type is 'dict' and it should be 'DataFrame'
        typecheck(1, int)
        typecheck(1.0, int)               # TypeError: Arg type is 'float' and it should be 'int'
        typecheck(1, str)                 # TypeError: Arg type is 'int' and it should be 'str'
        """
        if not isinstance(obj, t):
            raise TypeError(f"Arg type is '{type(obj).__name__}' and it should be '{t.__name__}'")
    
    @staticmethod
    def _check_dupes(df: pd.DataFrame, groups: list = ['cc_idnt', 'channel_num', 'week_start_date']):
        """
        Check a df for duplicates for each column in groups. This function is used to check data before they are passed into a model. 
        The input dataframe df should be unique for all groups. For example, the df does not a CC with several channels,
        or several CCs in a dept, or duplicate weeks.

        Examples
        --------
        tmp = pd.DataFrame({'cc_idnt':['123','123','123','123','123','123'], 
                'channel_num':['110','110','110', '120','120','120'], 
                'week_start_date':['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-01','2023-01-08','2023-01-15']})
        tmp 
        #       cc_idnt	    channel_num	    week_start_date	    demand
        # 0	    123	        110	            2023-01-01	        1
        # 1	    123	        110	            2023-01-08	        2
        # 2	    123	        110	            2023-01-15	        3
        # 3	    123	        120	            2023-01-01	        4
        # 4	    123	        120	            2023-01-08	        5
        # 5	    123	        120	            2023-01-15	        6

        hp.check_dupes(df=tmp)
        # AssertionError: df['channel_num'].unique()==2, should be 1.

        hp.check_dupes(df=tmp.loc[tmp['channel_num']=='110'])
        # returns None
        """
        self._typecheck(obj=df, t=pd.DataFrame)
        n = df.groupby(groups).count()['demand'].max()
        assert n == 1, f"N={n} rows found per group: {groups}"
        tmp0 = df[groups].nunique()
        for i in [i for i in ['week_start_date', 'channel_num', 'cc_idnt'] if i not in ['week_start_date', 'week_num', 'date']]:
            nu = tmp0[i]
            assert nu == 1, f"df['{i}'].unique()=={nu}, should be 1."

    # ABOVE = General helper functions
    # BELOW = Functions for inputs

    def _read_dept(self, dept_num, s3_key='input'):
        """ Wrapper around DSConfig()._read_parquet() to get department data.
        Note: Partitions are read as categorical variables which blow up memory 
        when cross-joined, so here dept_num dtype is changed to int.

        Example of data partition:
        s3://item-demand-forecast-nonprod/input/master-input-python/channel_brand=NORDSTROM/dept_num=806/selling_channel=ONLINE/week_num=202232/

        Dependencies
        ------------
        DSConfig()._read_parquet()

        Parameters
        ----------
        dept_num : int or str
            Department number, ex. 828
        s3_key : str
            from config.yaml; S3 input key ('input' = IDF master table)

        Examples
        --------
        df = self._read_parquet(dept_num=853)
        """
        # make path from inputted s3 key
        path = self.vars['s3_key'][s3_key]

        df = pd.DataFrame()
        for i in ['NORDSTROM', 'NORDSTROM_RACK']:
            tmp = self._read_parquet(bucket=self.s3, 
                    key=path + f'/channel_brand={i}', 
                    partitions={'dept_num':dept_num})
            df = pd.concat([df, tmp])
        
        df['dept_num'] = str(dept_num)

        # Explicitly set datatypes
        df = df.astype(self.actuals_dtypes)
        return df 
    
    @staticmethod
    def _astypes_wrapper(df: pd.DataFrame, dtypes: dict):
        """ Use astypes on available columns.
        Avoids the error: KeyError: 'Only a column name can be used for the key in a dtype mappings argument.'
        Useful for when we want to reuse a single dtype dict.
        """
        dtypes1 = {k: v for k, v in dtypes.items() if k in df.columns}
        return df.astype(dtypes1)

    # TODO: REPLACE THIS FUNCTION with more sensible join
    def _full_data(self,
                   df: pd.DataFrame, 
                   merch_cols: list=['channel_num', 'channel_brand', 'selling_channel','dept_num','epm_choice_num', 'class_num', 'brand_name']
                  ) -> pd.DataFrame:
        """ Cross join merch cols x time to get all rows. Join the raw data back to these full merch x time data.
        Nulls need to be filled after this join.

        # NOTE: This was simplified for Hackathon but it needs to be changed from cartesian join to the min/max for each CC since each CC had a run

        Dependencies
        ------------
        DSConfig._cartesian_product_multi(), dates

        Parameters
        ----------
        df : raw data.
        merch_cols : columns for which to find unique values.
        """
        df_dts = self.dates[['week_num']].drop_duplicates().sort_values(by=['week_num'])

        start = df['week_num'].min()
        end =   df['week_num'].max()
        df_dts = df_dts[df_dts['week_num'].between(start,end)]
        df_merch = df[merch_cols].drop_duplicates()

        df_out = self._cartesian_product_multi(df_merch, df_dts)
        df_out = df_out.merge(df, how='left')
        return df_out

    def _split_training_data(self, df, fcst_dt):
        """ Filter training data to dates < fcst_dt.

        Parameters
        ----------
        df : pandas.DataFrame
            training data with column 'week_start_date' (datetime64[ns])
        fcst_dt : datetime.date (or str)
            Forecast date. Most of the time our dates are stored as datetime.date,
            but string as 'YYYY-MM-DD' will also work (less preferred because of consistency).

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        df = pd.DataFrame({'week_start_date':['2022-01-01', '2022-02-04', '2022-03-06', '2022-04-10'],
                    'demand':[100, 200, 300, 400]}).astype({'week_start_date':'datetime64'})
        self._split_training_data(df=df, fcst_dt='2022-02-20')

        #       week_start_date	    demand
        # 0	    2022-01-01	        100
        # 1	    2022-02-04	        200
        """
        return df.loc[df["week_start_date"] < pd.Timestamp(fcst_dt)]

    @staticmethod
    def _create_pseudoCC_col(df: pd.DataFrame, d: str='dept_num', c: str='class_num', s: str=None, b: str='brand_name', clr: str=None)-> np.ndarray:
        """ Make pseudoCC from available columns
        
        Parameters
        ----------
        df : data with pseudoCC columns like dept, class, supplier...
        d,c,s,b,clr : columns for dept, class, subclass, brand, color respectively

        Examples
        --------
        my_df['cc_idnt'] = _create_pseudoCC_col(df=my_df)
        """
        def coz(col):
            " Return column or zero (coz)"
            try:
                return df[col].astype(str)
            except KeyError:
                return '0'
        return coz(d)+"~~"+coz(c)+"~~"+coz(s)+"~~"+coz(b)+"~~"+coz(clr)

    def _rank_filter(self,
                    df, 
                    groups=['dept_num'],
                    category='epm_choice_num',
                    value='demand',
                    start=None, 
                    end=None, 
                    dmas=None, 
                    top=None, 
                    bottom=None,
                    method='proportion'):
        """ Find top-ranked (or bottom-ranked) of a category according to value.
        There are 3 ranking methods:
            1. Top x %. Ex. return top 10% values.
            1. Top X rank. Return a fixed number of results. ex. return top 10.
            1. Top X cumulative proportion. Ex. return the results that compose 10% of the total value.
               This is like the Pareto principle, where 20% of a cohort are 
               generally responsible for 80% of results.
        
        Note: Dataframe can be filtered by start, end, and dmas.
        
        Parameters
        ----------
        df : pandas.DataFrame
            forecast dataframe
        groups : list(str)
            Find top X category for each of these group columns.
        category : str
            Column to check for top value. ex. if 'epm_choice_num', what are top-selling epm_choice_num
        start, end : str
            Start and end dates of forecast as YYYY-MM-DD. Dates are inclusive.
        dmas : list(num)
            Optional list of DMAs to filter
        top : num
            Top X. See method to see what "top" means.
        method : str {'percent', 'rank', 'cumulative proportion'}, default 'proportion'
            Return the top X, ex. if x=0.1 or 10:
            * percent: top 10% value calculated from the quantile.
            * rank: top 10 value calculated from ranking value.
            * cumulative proportion: top 10% of value from cumulative sum/total.
            Note: only 'rank' and 'cumulative proportion' are considered and any other string
            is evaluated as 'percent'.
        
        Returns
        -------
        list
            List of category values.

        Examples
        --------
        import numpy as np
        np.random.seed(123)
        N=100
        demo = pd.concat([
            pd.DataFrame({'dept':'A', 
                'CC': np.random.choice(np.arange(N)+1, replace=False, size=N),
                'sales': np.random.randint(0, 10000, 100)
                #'sales': np.round(np.random.gamma(1,1,N)*3000)
                })
            , pd.DataFrame({'dept':'B', 
                'CC': np.random.choice(np.arange(N)+1+N, replace=False, size=N),
                'sales': np.random.randint(0, 10000, 100)
                #'sales': np.round(np.random.gamma(1,1,N)*3000)
                })
        ])
        # View values sorted by top- and bottom-ranked
        df_top = demo.sort_values(by=['sales'], ascending=False)
        df_top['rank'] = df_top.groupby(['dept'])['sales'].rank(ascending=False)
        df_top['cp']= round(df_top.groupby(['dept'])['sales'].cumsum()/df_top['sales'].sum(),3)
        df_top['pct']= df_top.groupby(['dept'])['sales'].rank(ascending=False, pct=True)
        df_top.sort_values(by=['sales'], ascending=False)

        #       dept    CC      sale    rank    cp      pct
        # 9     B       168     9986    1.0     0.010   0.01
        # 79    A       40      9956    1.0     0.010   0.01
        # 27	B       103     9952    2.0     0.021   0.02
        # 36	A       96      9713    2.0     0.020   0.02
        # 99	A       67      9705    3.0     0.030   0.03
        # 3     A       29      9635    4.0     0.040   0.04
        # 74	B       136     9617    4.0     0.040   0.04
        # ...	...     ...     ...     ...     ...     ...
        # 8     A       82      321     99.0    0.499   0.99
        # 85    A       81      194     100.0   0.500   1.00
        # 40	B       143     191     98.0    0.500   0.98
        # 44	B       126     113     99.0    0.500   0.99
        # 33	B       113     16      100.0   0.500   1.00

        _rank_filter(df=demo, groups=['dept'], category='CC', value='sales', top=.03, method='proportion') # [168, 40, 103, 96, 67, 176]
        _rank_filter(df=demo, groups=['dept'], category='CC', value='sales', top=3, method='rank') # [168, 40, 103, 96, 67, 176]
        _rank_filter(df=demo, groups=None,     category='CC', value='sales', top=3, method='rank') # [168, 40, 103]
        _rank_filter(df=demo, groups=['dept'], category='CC', value='sales', top=.03, method='cumulative proportion') # [168, 40, 103, 96]

        # Run same code as df_top except with ascending=True to see bottom-ranked first
        _rank_filter(df=demo, groups=['dept'], category='CC', value='sales', bottom=.03, method='proportion') # [113, 126, 143, 81, 82, 30]
        _rank_filter(df=demo, groups=['dept'], category='CC', value='sales', bottom=3, method='rank') # [113, 126, 143, 81, 82, 30]
        _rank_filter(df=demo, groups=['dept'], category='CC', value='sales', bottom=.01, method='cumulative proportion') # [113,126,143,81,82,30,11,87,53,144,120,68,97,13,129,1,154,109,90,134,169,47,133,39]
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("No DataFrame to filter")
        if top and bottom:
            raise ValueError(f"Cannot enter both a top (={top}) and bottom(={bottom}) value.")
        if not top and not bottom:
            top=10  # default if None passed to both
        threshold = top or bottom   # limit
        if top:
            fl_asc=False
        if bottom:
            fl_asc=True

        # Before calculating top/bottom X, filter data.
        query=f"{value} != None" # generic placeholder
        if start:
            query = query + f" and week_start_date >= {str(start)}"
        if end:
            query = query + f" and week_start_date <= {str(end)}"
        if dmas:
            query = query + f" and dma_cd in {dmas}"

        # Filter and sort.
        # Note: The sort order affects results. ex. if calculating top sales,
        # then sort descending. rank, cumulative proportion, and percentile should
        # start from 0 or 1 for the highest sales value, then increase as sales decrease.
        # For bottom sales, sort ascending and all metrics increase with increasing sales.
        df1 = df.query(query)
        if groups is None:
            df1['tempFakeGroupByColumn']=1
            groups = ['tempFakeGroupByColumn']
        df1 = (df1.groupby(groups+[category])[value].
            sum().reset_index().
            sort_values(by=[value], ascending=[fl_asc]).reset_index(drop=True).
            astype({value:'float'})
        )

        # # Optional fields to see calculations altogether.
        # df1['rank'] = df1.groupby(groups)[value].rank(ascending=fl_asc)
        # df1['cp'] = df1.groupby(groups)[value].cumsum()/df1[value].sum()
        # df1['pct'] = df1.groupby(groups)[value].rank(ascending=fl_asc, pct=True)

        if method != 'rank':
            if (threshold > 1) or threshold < 0:
                raise TypeError(f"proportion should be [0,1]. You entered {threshold}")
        if method == 'rank':
            if threshold<1:
                raise TypeError(f"rank should be int => 1. You entered {threshold}")
            df1['result'] = df1.groupby(groups)[value].rank(ascending=fl_asc)
        elif method=='cumulative proportion':
            df1['result'] = df1.groupby(groups)[value].cumsum()/df1[value].sum()
        else:
            df1['result'] = df1.groupby(groups)[value].rank(ascending=fl_asc, pct=True)
        df1 = df1.drop(columns=df1.columns.intersection(['tempFakeGroupByColumn']))
        return list(df1.loc[df1['result'] <= threshold][category])

    @staticmethod
    def _ratio_by_group(df, by, sd=False):
        """ Calculate ratios by group (from https://stackoverflow.com/a/50296722/4718512).
        If sd, calculate standard deviation of ratios.
        These are not returned together so users can cleanly manage column renaming outside of this function.
        
        ex. group   value   ratio
            A       1       0.33
            A       2       0.67
            B       5       0.10
            B       10      0.20
            B       15      0.30
            B       20      0.40

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe with by columns
        by : list(str)
            Group columns
        sd : bool
            Calculate and return standard deviation instead of proportion

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        _ratio_by_group(df=pd.DataFrame({'group':['A','A','B','B','B','B'], 'value':[1,2,5,10,15,20]}), by='group')
        >  	value
        0 	0.333
        1 	0.667
        2 	0.100
        3 	0.200
        4 	0.300
        5 	0.400
        """
        groups = df.groupby(by)
        # computes group-wise sum, then auto broadcasts to size of group chunk
        summ = groups.transform(np.sum)
        p = (df[summ.columns]/summ)
        if sd:
            return np.sqrt(p*(1-p)/summ)  
        else:
            return p
    
    def _ratio(self, df, groups=None, target=None, sd=False):
        """ Wrapper around _ratio_by_group() that handles no groups (i.e. ratio over all rows) and formats output.
        Note: Proportions are calculated over all rows by groups, so aggregate data if necessary before passing the df to this function.
        ex. If the df has data by:
        dept | week | dma | units_sold
        If you want to calculate units sold by dept, aggregate units_sold by dept (sum over all weeks and dmas)
        If you want to calculate units sold by dept x week, aggregate units_sold by dept x week (sum over all dmas)

        Dependencies
        ------------
        _ratio_by_group()

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe with groups and target columns
        groups : list(str)
            Group columns
        target : list(str)
            Columns to calculate as proportions
        sd : bool
            Return standard deviation of proportions too?

        Returns
        -------
        pandas.DataFrame
        
        Examples
        --------
        df = pd.DataFrame({'group':['A','A','B','B','B','B'], 'value':[1,2,5,10,15,20]})
        _ratio(df=df, groups=['group'], target=['value'])
        >   group   value   p_value
        0   A       1       0.333
        1   A       2       0.667
        2   B       5       0.10
        3   B       10      0.20
        4   B       15      0.30
        5   B       20      0.40
        """
        num_cols = list(df.select_dtypes('number').columns)  # columns that can be calculated (avoids error in _ratio_by_group()
        target_cols = target or num_cols
        assert isinstance(target_cols,list)
        target_cols = list(set(num_cols).intersection(target_cols) - set(groups))

        df1 = df[groups + target_cols]
        #df1 = df[df.columns.intersection(groups + target_cols)]
        
        if groups is None:
            # then calculate ratio over all rows in df
            df1['dummy_group_col']=1
        
        ratios = self._ratio_by_group(df=df1, by=groups)
        ratios.columns = ['p_'+c for c in ratios.columns]

        if sd:
            sd_ratios = self._ratio_by_group(df=df1, by=groups, sd=True)
            sd_ratios.columns = ['psd_'+c for c in sd_ratios.columns]
            ratios = pd.concat([ratios, sd_ratios], axis=1)
        
        return df.merge(ratios, left_index=True, right_index=True)


    def _agg_with_rename(df, groups=['channel_num', 'dept_num', 'class_num', 'brand_name'] + ['week_num', 'week_start_date', 'week_of_fyr'],
                        n='epm_choice_num',
                        agg={'demand_sum': ('demand', 'sum'), 'demand_avg': ('demand', 'mean'), 'demand_sd': ('demand', 'std')},
                        qc_cols=False
                        ):
        """ Calculate the average and sd. This function is useful when the data 
        has many 0s that would drag down the average.
        Note1: the variable 'n' is used to calculate average because sales data
        may have lots of 0s for specific SKUs (fine resolution) that does not capture 
        broader level resolution like customer-choice (CC), and if these 0s were included
        in the average calculation, they would artificially lower the average of 
        the broader-resolution metric.
        See example demonstration for SKU- vs CC-based average.

        Note2: If a user enters the same column(s) in the 'groups' and 'n' args, 
        then those column(s) will be used in groups and ignored in n. For example, avg of
        CC sales by CC is the same as sum of sales by CC.

        Parameters
        ----------
        df : pandas.DataFrame
        groups : list(str)
            grouping columns.
        n : str
            Column over which to calculate average. Generally, this is epm_choice_num.
        agg : dict
            Dict of columns to aggregate in the format of:
            {'new_column_name':('column_A', 'function1'), 
            'another_new_column':('column_Z', 'function2'),
            ...}
            
        Returns
        -------
        pandas.DataFrame
            with aggregated values

        Examples
        --------
        # 3 CCs in the same dept x class. 2 x red, 1 x blue. 12 distinct skus
        > tmp = pd.DataFrame({
                'dept': [1,1,1,1,1,1,1,1,1,1,1,1],
                'class': [2,2,2,2,2,2,2,2,2,2,2,2],
                'color': ['red','red','red','red','red','red','red','red','red','red',  'blue','blue'],
                'CC': [987,987,987,987,987,987,  456,456,456,456,  321,321],
                'SKU': [98700,98701,98702,98703,98704,98705,  45600,45601,45602,45603, 32100,32101],
                'demand': [5,0,0,3,4,8,  0,2,0,6,   7,9]
        })
        > tmp  # df where every line is a unique sku
            dept    class   color   CC      SKU         demand
        0   1       2       red     987     98700       5
        1   1       2       red     987     98701       0
        2   1       2       red     987     98702       0
        3   1       2       red     987     98703       3
        4   1       2       red     987     98704       4
        5   1       2       red     987     98705       8
        6   1       2       red     456     45600       0
        7   1       2       red     456     45601       2
        8   1       2       red     456     45602       0
        9   1       2       red     456     45603       6
        10  1       2       blue    321     32100       7
        11  1       2       blue    321     32101       9
        # TODO: Redo example above since granularity must be CC, not SKU
        """
        # This logic allows users to enter the same column (ex. CC) in function args groups and n without erroring
        # Adding the same column to both args happens when looping over different merch hierarchies.
        # groups_for_avg = [i for i in list(set(groups+extra_date_cols)) if i in groups + extra_date_cols]
        groups_for_avg = [i for i in list(set(groups)) if i in groups]

        # Calculate sum over all groups
        agg_fns = {f'{n}_unique': (n, 'nunique'), f'{n}_count': (n, 'count')}
        stat_cols = list(agg_fns.keys())   # grab the stat column names for downstream col reordering
        agg_fns.update(agg)

        df1 = (df.groupby(groups_for_avg).
            agg(**agg_fns).
            reset_index())

        df1 = df1[groups_for_avg + stat_cols + list(agg.keys())]  # rearrange columns

        if qc_cols is False:
            df1.drop(columns=stat_cols, inplace=True)

        return df1


    # TODO: This function was written for sktime exploration
    def _aggregate_cols(df: pd.DataFrame, groups: list):
        """Wrap _agg_rename() on a bunch of columns.
        
        Dependencies
        ------------
        self._agg_with_rename()
        """
        # How are we going to aggregate the columns?
        # Specify the columns in lists here and we'll create the {new_column:(column,fn)} dict
        avg_cols = ['demand', 'demand_store_take', 'demand_owned', 'demand_dropship',
                    'boh', 'boh_sku_count', 'boh_store', 'boh_store_ct',
                    'regular_price_amt', 'current_price_amt', 'p_discount',
                    'allocated_qty', 'received_qty',
                    'averagerating', 'product_views', 'add_to_bag', 'rp_ind']
        sd_cols = ['demand']
        sum_cols = avg_cols  # ['rp_ind']                               # How many RP items for that pseudo_CC
        cnt_cols = ['regular_event_name', 'loyalty_event']  # count CCs with events

        # Trim columns down to those that are actually present
        list_of_cols = [df.columns.intersection(c).tolist() for c in
                        [avg_cols, sd_cols, sum_cols, cnt_cols]]

        # create dict of col names + agg funcions to pass to _agg_rename()
        # order of columns needs to match the values below
        cols_fn = zip(list_of_cols, ['mean', 'std', 'sum', 'count'])
        agg = {}
        for i in cols_fn:
            agg.update({c + '_' + i[1]: (c, i[1]) for c in i[0]})

        df1 = _agg_with_rename(df=df, groups=groups, n='epm_choice_num', agg=agg, qc_cols=True)
        # avgn(df1)  # calculate average as summed_col / unique epm_choice_num. These columns are named variable_avgn

        df1.columns = [c.replace('_mean', '') for c in df1.columns]

        return df1

    #### ABOVE = Functions for inputs
    #### BELOW = Functions for outputs
    @staticmethod
    def _zero_the_negatives(df: pd.DataFrame, nonzero_cols: list=['demand', 'y', 'y_low', 'y_high']) -> pd.DataFrame:
        """ Any vals < 0 are not valid forecasts because we forecast gross demand.
        
        Parameters
        ----------
        nonzero_cols : list of columns to non-zero.

        Examples
        --------
        fcst1 = zero_the_negatives(df=fcst)
        """
        colz = list(df.columns.intersection(nonzero_cols))
        df1 = df.copy()
        for c in colz:
            df1.loc[df1[c] < 0, c] = 0
        return df1

    def _forecast_info(self, df: pd.DataFrame=None, model: str=None, fcst_dt: str=None):
        """ Get different metadata parameters from an input dataframe:
            channel_num, selling_channel, channel_brand
            dept, class, supplier, color, cc_idnt
            model, fcst_dt, run_date

        Use this to process your forecast info and the output can be used with make_metadata() to generate forecast metadata.

        Dependencies
        ------------
        DSConfig: RUN_DATE_UTC
        """
        self._typecheck(df, pd.DataFrame)
        
        channelz = pd.DataFrame({'channel_num':['110', '120', '210', '250'], 
                'channel_brand':['NORDSTROM' ,'NORDSTROM', 'NORDSTROM_RACK', 'NORDSTROM_RACK'],
                'selling_channel': ['STORE', 'ONLINE', 'STORE', 'ONLINE']
                })
        info_colz = ['dept_num', 'class_num', 'brand_name', 'color_num', 'cc_idnt', 'channel_num',
            ]
        info_colz = list(df.columns.intersection(info_colz))
        output = (df[info_colz].drop_duplicates().
                merge(channelz).fillna('0').
                rename(columns={'dept_num':'dept', 'brand_name':'supplier'}))
        if 'color_num' not in info_colz:
            output['color_num']='0'

        output['model'] = model
        output['fcst_dt'] = fcst_dt
        output['run_date'] = self.RUN_DATE_UTC # YYYY-MM-DD
        assert output.shape[0]==1, f"forecast_info have n_rows= 1 but it is {output.shape[0]}."
        return output

    def _format_output(self, fcst: pd.DataFrame, info: pd.DataFrame, col_order: list = None):
        """ Combine forecast data with forecast metadata to get forecast output.
        1. Outer join forecast and metadata.
        2. Reorder columns.
        3. Format dates into objects for Parquet.
        """
        self._typecheck(fcst, pd.DataFrame)
        self._typecheck(info, pd.DataFrame)

        if not col_order:
            col_order = list(self.forecast_parquet_dtypes.keys())

        fcst1 = self._zero_the_negatives(df=fcst).rename(columns={'week_start_date':'date'})

        # join fcst and info dfs
        info1 = info.copy()
        fcst1['foo']=1
        info1['foo']=1
        # re-order columns
        output = fcst1.merge(info1, how='outer').drop(columns=['foo'])[col_order]

        # Convert datetime64[ns] to date objects
        output = self._parquet_dates(df=output)
        return output

    # # TODO: Deprecate or delete
    # def _make_metadata(df: pd.DataFrame, start, end, envi, git, ram, cpu):
    #     """ Make metadata from formatted forcast output, adding additional required metadata.
    #     see: https://confluence.nordstrom.com/pages/viewpage.action?pageId=925337071
    #     """
    #     df1 = df[['dept','run_date','fcst_dt','model','channel_brand','selling_channel','channel_country']].drop_duplicates()
    #     df1[['start_run_time','end_run_time','environment','git','memory','CPUs']]=start,end,envi,git,ram,cpu
    #     return df1

    def _parquet_dates(df: pd.DataFrame) -> pd.DataFrame:
        """ Convert any datetime64[ns] columns to dt.date.
        
        Examples
        --------
        df_out = _parquet_dates(df=df_in)
        """
        cols_to_change = list(df.dtypes[df.dtypes=='datetime64[ns]'].index)
        df1 = df.copy()
        for d in cols_to_change:
            df1[d] = df1[d].dt.date
        return df1

    def _format(self, 
            df, 
            fcst_dt=None, 
            run_date=None, 
            git=None, 
            model=None, 
            model_version=None, 
            partition_cols=['dept', 'model', 'fcst_dt'],
            no_negatives=True):
        """ Modify columns' names, order, and data types so the outputs are consistent.
        Add metadata to the dateframe. If metadata fields are None, use self.<variable> instead.
        Most of the time, only fcst_dt will need to be modified for the autoselection forecast.

        Note: this function drops the subclass column

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe with output columns
        fcst_dt, run_date : str
            forecast date and run date as YYYY-MM-DD
        git : str
            git describe
        model, model_version : str
            The model and version. If autoselection, then. model_chosen = original source model.
        partition_cols : list(str)
            Name of partition columns. These will be moved to the end of the dataframe in the order listed.
            This ordering facilitates writing Parquet files to S3 partitions.
        no_negatives : bool
            Set all negative values to 0?

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        # Set fcst_dt column for forecast and autoselection forecasts
        forecast = self._format(df=forecast, fcst_dt=self.fcst_dt)
        autosel_fcst = self._format(df=autosel_fcst, fcst_dt=self.eval_dt)
        """
        df1 = df.rename(columns={
                #'epm_choice_num':'cc_idnt', # cc_idnt is the CC or pseudo-CC
                'pseudo_cc':'cc_idnt',
                'dept_num':'dept',
                'class_num':'class',
                'sbclass_num':'subclass',
                'brand_name':'supplier',
                'dma_cd':'dma',
                'dma_code':'dma',
                'y':'avg_demand',
                'y_lower':'low_demand',
                'y_upper':'high_demand',
                'week_start_date':'date'
            })

        # Add forecast information
        df1['fcst_dt'] = fcst_dt or self.fcst_dt
        df1['run_date'] = run_date or self.RUN_DATE_UTC
        df1['git'] = git or self.git
        df1['model'] = model or self.model 
        df1['model_version'] = model_version or self.model_version
        # TODO: integrate this with autoselection- need to populate for regular models and autoselction already sets this column.
        # The autoselection class will write the original model_chosen, and all other models will use this placeholder logic.
        if model != 'autoselection':
            df1['model_chosen'] = None

        if no_negatives:
            df1[['avg_demand', 'low_demand', 'high_demand']] = df1[['avg_demand', 'low_demand', 'high_demand']].clip(lower=0)

        # final columns and order to submit for S3 upload
        col_order = ['channel_brand', 'selling_channel', 'channel_country',
                    'dept', 'class', #'subclass', 
                    'supplier', 'color_num', 'cc_idnt', 
                    'date', 'dma',  
                    'avg_demand', 'low_demand', 'high_demand',
                    'fcst_dt', 'run_date', 'git', 
                    'model', 'model_version', 'model_chosen']

        # Explicitly set column type.
        # Note: for Hive tables built on the S3 data, date strings or datetimes
        # may need to be converted to date objects.
        df1 = df1.astype({
                "channel_brand": str,
                "selling_channel": str,
                "channel_country": str,
                "dept": int, 
                "class": int, 
                #"subclass": int,
                "supplier": str, 
                "color_num": str,
                "cc_idnt": str, 
                "date": str, #TODO: 'datetime64[ns]',
                "dma": int, 
                "avg_demand": float, 
                "low_demand": float, 
                "high_demand": float,
                "fcst_dt": str,  #TODO: 'datetime64[ns]',
                "run_date": str, #TODO: 'datetime64[ns]',
                "git": str,
                "model": str,
                "model_version": str,
                "model_chosen": str
            })

        # Move partition columns to the end so that a Hive table can be overlayed on S3 data
        try:
            df1 = df1[col_order]
        except KeyError as e:
            self.LOG.exception(f"Check that data columns:\n{df1.columns.tolist()}\nmatch required columns:\{col_order}")
            raise e
        df1 = df1[[c for c in df1.columns.tolist() if c not in partition_cols] + partition_cols]
        
        return df1

    @staticmethod
    def _check_unique(df, col):
        """This function checks to make sure a dataframe only has one unique value in a column.
        This is used for setting metadata or partition names from a dataframe.

        ex. Forecasts in a training run should not have multiple as-of forecast dates, and
        multiple dates in a dataframe will cause an AssertionError.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to check
        col : str
            name of column in dataframe to check.

        Returns
        -------
        value matching datatype of df[col]

        Examples
        --------
        df = pd.DataFrame({'model':['foo', 'foo', 'foo'],
                    'dept':[100, 100, 200]})
        self._check_unique(df=df, col='model')
        > 'foo'
        self._check_unique(df=df, col='dept')  # duplicates
        > AssertionError
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"No dataframe passed. type(df)=f{type(df)}")
        vals = df[col].unique()
        assert len(vals) == 1, f"len(unique {col})=={len(vals)}. df['{col}'].unique()={vals}."
        return vals[0]

    def _check_percent_missing(self, df):
        """This function checks if a given dataframe has any missing values. 
        If yes, columns and percent missing is logged. 

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe to check

        Examples
        --------
        df = pd.DataFrame({'model':['foo', 'foo', 'foo'],
                    'dept':[100, 100, np.nan]})
        self._check_percent_missing(df=df)
        > Columns with missing data: {'dept': 33.333333333333336}
        """
        percent_missing = pd.DataFrame(df.isnull().sum() * 100 / len(df))
        percent_missing.columns = ['percent_missing']
        missing = percent_missing['percent_missing'][percent_missing['percent_missing']>0].to_dict()
        if len(missing) > 0: 
            self.LOG.info(f'Columns with missing data: {missing}')
        else: 
            self.LOG.info(f'No columns have missing data')
        return missing

    def _upload_forecast_wrapper(self, df=None, df_sel=None, staged=False, overwrite=False):
        """ Upload a forecast to S3 using wrapper around awswrangler.s3.to_parquet().
        This wrapper facilitates uploads to s3 keys archive and tempstaged.
        A follow-up job copies all of the tempstaged to staged to minimize
        odds of disruption to forecast-lookup.

        Data are stored with partitions like:
        archive/
        |-dept=123/model=FancyModel/fcst_dt=2022-07-27
          |- NCOM_a5f461e1d94d4eac8149040a87e03671.snappy.parquet
          |- RCOM_8c0b8ce97c1a409d80e79d0b1584c6e7.snappy.parquet
        temp_staged/
        |-dept=123/
          |- NCOM_a5f461e1d94d4eac8149040a87e03671.snappy.parquet
          |- RCOM_8c0b8ce97c1a409d80e79d0b1584c6e7.snappy.parquet
        so the first channel uploaded can clear out the s3 bucket with overwrite=True
        (-or- explicitly use DSConfig._clear_s3() first),
        and subsequent channels can use overwrite=False.

        Note: Data can be added to s3 without replacing existing data if overwrite=False, 
        so duplication may occur for the same channel_cols x cc_idnt x fcst_dt x run_date x model x model_version x model_chosen x git.
        Debug this likely rare event by comparing timestamps of Parquet files in S3.

        Dependencies
        ------------
        self.model
            Instance should have the model name. This is used for the partition.
            self.model is used because autoselection will be the Object.model for the autoselection instance
            even though the instance will be handling dataframes with other model names.
        DSConfig: _upload() 
            Generic upload function for pandas.DataFrame to S3 (CSV or Parquet). Uses awswrangler.
        DSConfig: self.LOG, self.s3, self.vars
        self.fcst_dt, self.eval_dt

        Parameters
        ----------
        df, df_sel : pandas.DataFrame
            Forecasts to upload. 
            ** The fcst_dt column should have the correct forecast date for each df. **
            df = forecast as of forecast date
            df_sel = shorter forecast made for auto-selection window.
        staged : bool
            Upload forecast to the staged s3 key? 
            False for models, and True for autoselection final model.
        overwrite : bool
            Overwrite any existing data in that s3 partition path before writing results?

        Returns
        -------
        Nothing. Data uploaded to S3.

        Examples
        --------
        # Upload a forecast
        self._upload_forecast_wrapper(df=forecast, df_sel=autoselection_fcst, staged=False, overwrite=True)
        # Upload the autoselected model winner
        self._upload_forecast_wrapper(df=forecast, df_sel=None, staged=True, overwrite=True)
        """
        assert isinstance(self.model, str), f"self.model={self.model}. Needs to be a string."

        # Mini-helper functions to determine partitions and file prefix
        def __partitions(df):
            """Create partitions from the dataframe.
            Dataframe should only have a single department and forecast date when _upload(overwrite=True).
            """
            assert isinstance(df, pd.DataFrame), f"type(df)={type(df)}. Needs to be a pandas.DataFrame"
            partitions = {'dept': self._check_unique(df=df, col='dept'), # make sure function iterates over depts
                          'model': self.model, 
                          'fcst_dt': self._check_unique(df=df, col='fcst_dt')}
            return partitions
        def __chan_lookup(df):
            """Deterimine the channel filename prefix from the dataframe".
            Dataframe should only have a single channel_brand x selling_channel, channel_country."""
            
            chan_dict = {i: self._check_unique(df=df, col=i) for i in ['channel_brand', 'selling_channel', 'channel_country']}
            return self._parse_channel(d=chan_dict)
        # Determine channel from data. This will be file prefix
        # Additionally provide S3 bucket-owner control to all files that are written by our NSK IAM role #
        pkwarg = {'filename_prefix':__chan_lookup(df=df)+'_',}
        
        # Upload to archive
        if isinstance(df, pd.DataFrame):
            self._upload(df=df, bucket=self.s3,
                        key=f"{self.vars['s3_key']['archive_forecasts']}",
                        partitions=__partitions(df=df), 
                        parquet_kwargs=pkwarg,
                        overwrite=overwrite)
        
        # Upload to autoselection s3 if df_sel=pd.DataFrame
        if isinstance(df_sel, pd.DataFrame):
            self._upload(df=df_sel, bucket=self.s3, 
                        key=f"{self.vars['s3_key']['evaluation_output']}", 
                        partitions=__partitions(df=df_sel), 
                        parquet_kwargs=pkwarg,
                        overwrite=overwrite)
        
        # Upload to staged folder (for most-recent weekly production jobs)
        if staged:
            partitions3 = {'dept': self._check_unique(df=df, col='dept')}
            self._upload(df=df, bucket=self.s3, 
                        key=f"{self.vars['s3_key']['tempstaged_forecasts']}", 
                        partitions=partitions3, 
                        parquet_kwargs=pkwarg,
                        overwrite=overwrite)

        self.LOG.info("Upload(s) finished.")
   
    @staticmethod
    def _get_all_s3_objects(s3, **base_kwargs):
        """Get all objects from S3, even if object count > 1000.
        https://stackoverflow.com/a/54314628/4718512
        
        Parameters
        ----------
        s3 : boto3.client('s3')
        base_kwargs : dict
            Args to pass to boto3.client.list_objects_v2
            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2

        Returns
        -------
        generator
        """
        continuation_token = None
        while True:
            list_kwargs = dict(MaxKeys=1000, **base_kwargs)
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token
            response = s3.list_objects_v2(**list_kwargs)
            yield from response.get('Contents', [])
            if not response.get('IsTruncated'):  # At the end of the list?
                break
            continuation_token = response.get('NextContinuationToken')


    def _move_s3_data(self, bucket=None, FROM=None, TO=None, char=4, move=True):
        """ Copy data FROM one s3 key TO another key.

        RUN THIS THROUGH A SINGLE WORKER because it copies everything and deletes the FROM key data.
        Make sure each key only has one file.
        Data in the FROM key will replace data in the TO key. 
        Data in the FROM key that are new will move to the TO key.
        Data in the TO key that do not exist in the FROM key are not deleted.

        WARNING: boto3 only allows 1000 keys at a time. This function can accommodate >1000 keys.
        Long-term replace this with aws (see below):
        aws s3 --recursive mv s3://<bucketname>/<folder_name_from> s3://<bucket>/<folder_name_to>
        but we don't have aws installed in the docker image.

        Dependencies
        ------------
        DSConfig: s3, LOG

        Parameters
        ----------
        bucket : str
            s3 bucket
        FROM, TO : str
            s3 keys FROM which TO move data. Should end in a '/'. ex. 'source/'
        move: bool
            If true, move files (copy to destination, delete from source).
            If false, copy files only (copy to destination, leave source files).
        char: int
            length of characters to look for in the file name after dept_num=X/....
            4 = "NCOM_" / "RCOM_"

        Returns
        -------
        Nothing. Data moved in S3.

        Examples
        --------
        self._move_s3_data(FROM='temporary_key/', TO='final_key/', move=True)
        """
        if not isinstance(bucket, str):
            bucket = self.s3

        if (not isinstance(FROM, str)) & (not isinstance(TO, str)):
            raise KeyError(f"FROM ({type(FROM)}) and TO ({type(TO)}) must be str.")
        for i in [FROM, TO]:
            if i[-1] != '/':
                raise ValueError(f"FROM={FROM} and TO={TO} must end in '/'.")

        s3 = boto3.resource('s3')

        def _s3_copier(s3_file, delete=move):
            copy_source = {'Bucket': bucket, 'Key': s3_file}
            new_s3_file = TO+s3_file[len(FROM):] # replace source key prefix with TO key prefix
            s3.meta.client.copy(CopySource=copy_source, Bucket=bucket, Key=new_s3_file)
            if delete:
                s3.meta.client.delete_object(Bucket=bucket, Key=s3_file)
        
        def get_files(key):
            yield from [f['Key'] for f in self._get_all_s3_objects(s3=boto3.client('s3'), Bucket=bucket, Prefix=key)]

        ls_to_files = list(get_files(TO))
        pre_files_to = [k.rsplit('/',1)[0].rsplit('/',1)[1] for k in ls_to_files]
        suff_files_to = [k.rsplit('/',1)[1][:char+1] for k in ls_to_files]
        TO_files = [m+ "/" + str(n) for m,n in zip(pre_files_to,suff_files_to)]
    
        
        ls_from_files = list(get_files(FROM))
        pre_files = [k.rsplit('/',1)[0].rsplit('/',1)[1] for k in ls_from_files]
        suff_files = [k.rsplit('/',1)[1][:char+1] for k in ls_from_files]
        FROM_files = [m+ "/" + str(n) for m,n in zip(pre_files,suff_files)]
        
       
        # Find partitions that exist in both TO and FROM keys. 
        eligible_for_deletion = set(TO_files) & set(FROM_files)
        eligible_for_deletion = [TO+i for i in eligible_for_deletion]

        def search_files(list_strings, search_term):
            files= []
            for s in list_strings:
                for x in search_term:
                    if re.search(x, s):
                        files.append(s)
                    else:
                        continue
            return files
        
        eligible_files = search_files(ls_to_files, eligible_for_deletion)
        self.LOG.info(f"delete files: {eligible_files}")
        # then delete the old files in the TO keys
        
        self.LOG.info(f"Clear {len(eligible_files)} files from destination: {bucket}/{TO}")
        for f in eligible_files:
            s3.meta.client.delete_object(Bucket=bucket, Key=f)
        
         # Now copy files FROM source
        self.LOG.info(f"Copy {len(ls_from_files)} files from {bucket}/{FROM} to {TO}")
        for f in ls_from_files:
            _s3_copier(f)
        
        self.LOG.info(f"temp to staged completed")

        
        
        
    def _copy_forecasts_to_staged(self):
        """ Copy all of the forecasts in temp_staged/ to staged/ to minimize
        odds of disruption to forecast-lookup.

        RUN THIS SEPARATELY THROUGH A SINGLE WORKER because it copies everything and
        deletes the old temp_staged key all at once.
        # TODO: Change this to 1. check what is in tempstaged, 2. if the same prefix is in staged, delete the staged prefix, 3. copy files from tempstaged-->staged.
        # TODO: This will allow us to replace staged files if a fresher result exists, and keep older ones. If staged has no tempstaged equivalent, ex. dept has results for last week but not this week, then those results are not deleted.
        #TODO: We should also have an arg for explicit depts, ex. tempstaged/dept=X --> staged/dept=X

        WARNING: boto3 only allows 1000 keys at a time. Long-term replace this with aws (see below)
        This is the same as:
        aws s3 --recursive mv s3://<bucketname>/<folder_name_from> s3://<bucket>/<folder_name_to>
        but we don't have aws installed in the docker image.

        Dependencies
        ------------
        DSConfig self.: s3, vars, LOG

        Parameters
        ----------
        None

        Returns
        -------
        Nothing. Data moved in S3.

        Examples
        --------
        self._copy_forecasts_to_staged()
        # tempstaged/ data should move to and replace staged/ data
        """
        s3 = boto3.resource('s3')
        # Note: add forward slash to keys
        source = f"{self.vars['s3_key']['tempstaged_forecasts']}/"
       
        dest = f"{self.vars['s3_key']['staged_forecasts']}/"

     
        def _s3_copier(s3_file):
            copy_source = {'Bucket': self.s3, 'Key': s3_file}
            new_s3_file = dest+s3_file[len(source):] # replace source key prefix with dest key prefix
            s3.meta.client.copy(CopySource=copy_source, Bucket=self.s3, Key=new_s3_file)
            s3.meta.client.delete_object(Bucket=self.s3, Key=s3_file)
        
        #TODO: Fix this to check temp-staged first for what's available and only delete corresponding dept in staged
        #TODO: this will help in case we do a few depts at a time.
        # Delete current files in staged
        while True:
            try:
                old_objs = s3.meta.client.list_objects_v2(Bucket=self.s3, Prefix=dest)
                old_files = [o['Key'] for o in old_objs['Contents']]
                self.LOG.info(f"Clear {len(old_files)} from destination: {dest}")
                for f in old_files:
                    s3.meta.client.delete_object(Bucket=self.s3, Key=f)
            except KeyError:
                break                    

        # In case we surpass 1000 files, repeat until no files left.
        while True:
            try:
                objs = s3.meta.client.list_objects_v2(Bucket=self.s3, Prefix=source)
                files = [o['Key'] for o in objs['Contents']]
                if len(files) == 1000:
                    self.LOG.warning("Not all files may have been moved because of boto3's 1000 file limit.")
                self.LOG.info(f"Moving {len(files)} files from {source} to {dest}")
                for f in files:
                    _s3_copier(f)
            except KeyError:
                break

        # Parallelize does not work because can't pickle boto3. Might work outside of class.
        # # Parallelize copy
        # from multiprocessing import Pool
        # # copy n objects at the same time
        # with Pool(processes) as p:  # TODO: document what 'processes' should be
        #     p.map(_s3_copier, files)
