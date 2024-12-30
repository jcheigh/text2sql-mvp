import os
import re 

def find_path(basename='text2sql-mvp'):
    path = os.getcwd()  
    while True:
        if os.path.basename(path) == basename:
            return path
        elif path == os.path.dirname(path):
            raise FileNotFoundError(f"No {basename} directory found above current path.")
        path = os.path.dirname(path)

def get_paths():
    MAIN_PATH    = find_path()
    SRC_PATH     = os.path.join(MAIN_PATH, "src")
    DATA_PATH    = os.path.join(MAIN_PATH, "data")
    ENV_FPATH    = f'{DATA_PATH}/keys.env'
    SQL_DB_FPATH = f'{DATA_PATH}/synthetic_data.db'

    return {
        "main" : MAIN_PATH,
        "src"  : SRC_PATH,
        "data" : DATA_PATH,
        "env"  : ENV_FPATH,
        "sql"  : SQL_DB_FPATH
        }

def get_schema_context(config, tbls_to_exclude=[]):
    db = config.db
    tables = db.get_usable_table_names()

    schema_lines = []
    for table in tables:
        if table not in tbls_to_exclude:
            table_info = db.get_table_info([table])
            schema_lines.append(f"Table: {table}\n{table_info}\n")

    schema_context = ("DATABASE SCHEMA:\n" + "\n".join(schema_lines))
    return schema_context

def extract_query(response, type='sql'):
    pattern = rf"```{type}\s+([\s\S]*?)\s+```"
    match   = re.search(pattern, response)
    
    if match:
        return match.group(1).strip()
    else:
        print(f"Extracting query of type {type} failed: returning response.strip():\n{response.strip()}")
        return response.strip()
    
