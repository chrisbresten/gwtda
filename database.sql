create schema gwcnn;
create table gwcnn.models (time_inserted timestamp default now(), id serial primary key, model_json json, filename text, modelhash char(32) not null, filehash char(32) not null, code text);
alter table gwcnn.models add constraint model_uniq unique (modelhash,filehash);

create table gwcnn.runs (time_inserted timestamp default now(), id serial primary key, cmdline_args text, test_size int, test_accuracy float, modelhash char(32), filehash char(32), savefile text, loadfile text, gitinfo text,notes text, params json, outputs jsonb, weights json);
create index on gwcnn.runs using btree(test_accuracy);
create index on gwcnn.runs using gin(to_tsvector('simple',cmdline_args));
create index on gwcnn.models using btree(modelhash);
alter table gwcnn.runs add column weightshash char(32);
alter table gwcnn.runs add constraint weights_uniq unique(weightshash);

