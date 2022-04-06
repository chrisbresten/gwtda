
GWTDA 0.0.1
+++++++++++
===========

This is not software intended for endusers. This is at most the begining of TDA time series classification module, or something of that natre.



userspace code, these are more scripts/examples of things being done:
- `signal_synthesis.py` : synthesis of GW signals from signal sources and noise sources, and construction of sliding window embedding
- `calc_tda.py` : generate the persistence diagrams of time series and store embedding thereof
- `viz_con.py` : visualization of activation and convolution
- `tda_CNN.py` : CNN classifier for TDA embedded features
- `run_CNN.py` : classifier with CNN on raw data and TDA embedded features combined


library-like code that does interesting and potentially useful things, in coherent organized form:
- `spectralnoise.py` : generates noise of observed PSD for synthesis and colored perterbation of 
- `embeddings.py` : persistence diagram vector space embeddings, sliding window embedding
- `logrun.py` : utility to log tensorflow results in postgres database for easy access for imaging

interfaces, convenience, generic hacks in this codebase:
- `ligoprep.py` : utility used to segment ligo data
- `signal_sources.py` : interface to conveniently access a few different signal sources
- `noise.py` : interface to access noise sources and use them
- `spsql.py` : utility for managing postgres connections, overkill for this sort of application but convenient for me. originally made for keep-alive features in more robust back-end applications
- `tools.py` : misc helper functions and hacks






userspace launch sequence:
```
python3 signal_synthesis.py ligo ligo 2000 
python3 calc_tda.py gw_*.npy 
python3 run_CNN.py tda_*.py
```






Storing results
===============

logrun.py interacts with a somewhat sophisticated system for saving and organizing results and code run with a postgres backend. This is meant to be optional and will not block execution of classifiers if broken. 

This is a utility designed for people hand tweaking tensorflow models, to keep track of every tensorflow model that is trained and tested, without the need for extra user input at runtime.  

it saves every model architecture, weights, test results, and code that made them, in a postgresql schema. This is done automatically so if the code is edited, it will automatically update a table, adding a row for a new model. 

There are 2 tables, one for models and one for runs of them. rows are added to the model table when the system sees a new model being used. the weights and results of every run, along with identifiers to the model that made them are stored in the runs table. .







install postgres
================
this is for reference to people who dont ever use databases

setting up the database from scratch. it is not hard, only thing is the auth, which can be a little weird if you are not used to databases. 

```
sudo apt install postgresql-12
```

then also install psycopg2 and dotenv

```
sudo apt install python3-psycopg2 python3-dotenv 
pip3 install tenacity
```

if you are on a single user machine or dont care about security you can add 
```
local   all             postgres                                trust
```

to `/etc/postgresql/12/main/pg_hba.conf`, then do `psql -U postgres` and 
```postgres=# \password 
Enter new password: 
```
you need to set the password to access it through psycopg, for the command line client it can do a trust based auth locally through a pipe, fifo,  emulated filesystem object thing. keep in mind that postgres is the admin account...

init the schema
```
createdb -U postgres gwtda
cat database.sql|psql -U postgres
```


put the auth info into the flne `.env`
