database.sql - schema to init the postgres(maybe works on other) sql database
embeddings.py - module to do TDA embeddings
example.env - example .env file
loadtda.py - load up signals and do sliding window and TDA stuff
logrun.py - module to save tensorflow runs
mk_noisy_signals.py - expand data by adding noise and shifting position
run_CNN.py - run CNN
spsql.py - code to manage database connection with some keep alive features


sequence:
```
python3 mk_noisy_signals.py gw
python3 loadtda.py gw_*.npy 
python3 run_CNN.py tda_*.py
```





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
