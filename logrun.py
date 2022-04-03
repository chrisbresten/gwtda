import json
import spsql
import __main__
import os
import hashlib

from dotenv import load_dotenv

load_dotenv(".env")
SCHEMA = os.getenv("SCHEMA", "gw")
try:
    filez = __main__.__file__
except:
    filez = "console"
uniquevilation = spsql.psycopg2.errors.UniqueViolation


class ModelLog:
    def __init__(self, model_json):
        if type(model_json) == dict:
            self.model_json = json.dumps(model_json)
        else:
            self.model_json = model_json
        self.s = spsql.spsql()
        self.filename = os.path.basename(filez)
        self.modelhash = hashlib.md5(self.model_json.encode("utf-8")).hexdigest()
        if filez != "console":
            b = open(filez)
            code = b.read()
            self.filehash = hashlib.md5(code.encode("utf-8")).hexdigest()
        else:
            code = ""
            self.filehash = hashlib.md5(filez.encode("utf-8")).hexdigest()
        try:
            self.s.curs.execute(
                "insert into "
                + SCHEMA
                + ".models (model_json, filename, filehash, modelhash,code) VALUES (%s,%s ,%s,%s,%s)",
                (self.model_json, self.filename, self.filehash, self.modelhash, code),
            )
        except uniquevilation:
            pass

    def logrun(
        self,
        accuracy,  # whatever you want to sort it by the easiest
        test_size,  # relevant to the above metric generally
        savefile,
        loadfile,
        notes="",
        params=dict(),
        output=dict(),
        weights=dict(),
    ):
        if type(params) == dict:
            params = json.dumps(params)
        if type(output) == dict:
            output = json.dumps(output)
        if type(weights) == dict:
            weights = json.dumps(weights)
            weightshash = hashlib.md5(weights.encode("utf-8")).hexdigest()
        else:
            weightshash = hashlib.md5((output + params).encode("utf-8")).hexdigest()
        # automatically get the cmdline args from the code at
        _cmdline_args = __main__.sys.argv
        cmdline_args = ""
        for a in _cmdline_args:
            cmdline_args = cmdline_args + " " + a
        print(f"weightshash:{weightshash}")
        self.s.curs.execute(
            "insert into "
            + SCHEMA
            + ".runs (cmdline_args, test_accuracy,test_size, savefile, loadfile,filehash, modelhash,gitinfo,notes, outputs, params,weights,weightshash) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                cmdline_args,
                accuracy,
                test_size,
                savefile,
                loadfile,
                self.filehash,
                self.modelhash,
                self.gitstuff,
                notes,
                output,
                params,
                weights,
                weightshash,
            ),
        )
        print(f"modelhash:{self.modelhash}")

    # pull info about the state of the git repo
    @property
    def gitstuff(self):
        try:
            sh = os.popen("git log --raw --max-count=1")
            out = sh.read()
            sh.close()
            return str(out)
        except:
            return ""
