
calc_tda.py
database.sql
dotda.sh
embeddings.py - embeddings of persistence diagrams
noise.py - noise helpers
run_CNN.py - hybrid CNN tda feature scheme
signal_sources.py - convenience code to load data sources
spectralnoise.py - 
spsql.py - helper function for postgres connection magnagement(overkill for this))
tda_CNN.py - CNN on only TDA features
tools.py - misc helpers and io stuff






sequence:
```
python3 mk_noisy_signals.py gw white 2000 
python3 calc_tda.py gw_*.npy 
python3 run_CNN.py tda_*.py
```



Storing results
===============

logruns.py interacts with a somewhat sophisticated system for saving and organizing results and code run with a postgres backend. This is optional and will not block code execution if broken

it saves every model architecture, weights, test results, and code that made them, in a postgresql schema. This is done automatically so if the code is edited, it will autpmatiocally update a table, adding a row for a new model. 

There are 2 tables, one for models and one for runs of them. rows are added to the model table when the system sees a new model being used. the weights and results of every run, along with identifiers to the model that made them are stored in the runs table. .







install postgres
================
setting up the database from scratch. it is not hard, only thing is the auth, which can be a little weird

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
you need ot set the password to access it throguh psycopg, for the command line client it can do a trust based auth locally through a pipe, fifo,  emulated filesystem object thing

init the schema
```
createdb -U postgres gwtda
cat database.sql|psql -U postgres
```


put the auth info into the fine `.env`
