import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from agent import Agent, Config
from helper import get_paths, extract_query, get_schema_context

class BasicAgent(Agent):

    def __init__(self, config: Config, log_data=True):
        super().__init__(config)
        self.log_data = log_data
        self.path_map = get_paths()  

    def _log(self, message: str):
        """
        Helper method to handle logging to both console and a log file.
        Appends each message as a new line.
        """
        print(message)
        if self.log_data:
            try:
                with open(self.path_map['log'], 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"[WARNING] Could not write to log file: {e}")

    def run(self, user_query: str) -> float:
        """
        Takes a user query string, runs it through:
          get_sql_json -> compile_sql -> get_python_prompt -> execute_python
        and returns a float result.
        """
        self._log("=" * 50)
        self._log("[STEP 0] Starting 'run' method.")
        self._log(f"User Query:\n{user_query}")

        start = time.time()

        # 1) Convert the user query to SQL JSON
        sql_json = self.get_sql_json(user_query)

        # 2) Compile SQL and fetch data into a dataframe
        df = self.compile_sql(sql_json)

        # 3) Generate a Python prompt describing the next steps
        py_prompt = self.get_python_prompt(user_query, df)

        # 4) Execute the described Python steps and retrieve the result
        result = self.execute_python(py_prompt, df)

        end = time.time()
        self._log("=" * 50)
        self._log(f"[STEP 5] Completed run in {round(end - start, 2)} seconds.\n")

        # Validate final result is float-like
        if not isinstance(result, (int, float)):
            raise ValueError(f"Python Execution Failed: the final result must be a float or int, instead got {type(result)}")
        
        return float(result)

    def validate_sql_json(self, sql_json: dict):
        """
        Validates a JSON structure that specifies:
        1) A list of tables with name, alias, and columns
        2) A list of joins describing how the tables are joined

        - Checks if the referenced tables and columns exist in the schema.
        - Checks if join column types match.
        - Joins must use one of the valid join types: {"inner", "left", "right", "outer"}.

        Returns:
            (True, "") if valid
            (False, <error_message>) if invalid
        """
        # 1) Basic structure checks
        if "tables" not in sql_json:
            return False, "Missing top-level 'tables' key."
        if "joins" not in sql_json:
            return False, "Missing top-level 'joins' key."

        # Gather known table aliases for join checks
        table_aliases = {}

        # 2) Validate each table + columns
        for tbl_obj in sql_json["tables"]:
            # required fields: name, alias, columns
            if "name" not in tbl_obj or "alias" not in tbl_obj or "columns" not in tbl_obj:
                return False, "Each table object must have 'name', 'alias', and 'columns'."

            tbl_name = tbl_obj["name"]
            tbl_alias = tbl_obj["alias"]
            if tbl_name not in self.schema:
                return False, f"Table '{tbl_name}' not found in schema."

            table_aliases[tbl_alias] = tbl_name

            # Validate columns
            for col in tbl_obj["columns"]:
                if "original_name" not in col:
                    return False, "Each column object must have 'original_name'."
                if "alias" not in col:
                    return False, "Each column object must have 'alias'."
                if "keep" not in col:
                    return False, "Each column object must have 'keep' (boolean)."

                col_name = col["original_name"]
                if col_name not in self.schema[tbl_name]["columns"]:
                    return False, f"Column '{col_name}' does not exist in table '{tbl_name}'."

        # 3) Validate joins
        valid_join_types = {"inner", "left", "right", "outer"}
        for j in sql_json["joins"]:
            # required fields: left_table_alias, right_table_alias, left_column, right_column, join_type
            required_keys = [
                "left_table_alias",
                "right_table_alias",
                "left_column",
                "right_column",
                "join_type"
            ]
            for k in required_keys:
                if k not in j:
                    return False, f"Join object must have '{k}'."

            lt_alias = j["left_table_alias"]
            rt_alias = j["right_table_alias"]
            lt_col = j["left_column"]
            rt_col = j["right_column"]
            jtype = j["join_type"]

            # Check table alias existence
            if lt_alias not in table_aliases:
                return False, f"Join references unknown table alias '{lt_alias}'."
            if rt_alias not in table_aliases:
                return False, f"Join references unknown table alias '{rt_alias}'."

            # Check valid join type
            if jtype not in valid_join_types:
                return False, f"Invalid join type '{jtype}'. Must be one of {valid_join_types}."

            # Check column existence in schema
            lt_table = table_aliases[lt_alias]
            rt_table = table_aliases[rt_alias]
            if lt_col not in self.schema[lt_table]["columns"]:
                return False, f"Column '{lt_col}' not found in table '{lt_table}'."
            if rt_col not in self.schema[rt_table]["columns"]:
                return False, f"Column '{rt_col}' not found in table '{rt_table}'."

            # Optional: check column types match
            lt_type = self.schema[lt_table]["columns"][lt_col]
            rt_type = self.schema[rt_table]["columns"][rt_col]
            if lt_type != rt_type:
                return False, (
                    f"Join column type mismatch: '{lt_table}.{lt_col}' is '{lt_type}' while "
                    f"'{rt_table}.{rt_col}' is '{rt_type}'."
                )

        # If everything is valid
        return True, ""

    def get_sql_json(self, user_query: str) -> dict:
        """
        Takes a user query and asks the LLM to produce a JSON specifying which tables/columns to pull.
        Validates the JSON structure and join types.
        """
        self._log("=" * 50)
        self._log("[STEP 1] Generating SQL JSON from user query...")

        schema_context = get_schema_context(self.config)
        prompt = f'''Given a user query and a SQLite database schema, return ONLY a valid JSON describing the 
        data required to answer the user query. The JSON should be parsable and adhere to proper JSON syntax.

        Instructions:
        2. Ensure column names and new column names are strings, and there are no extraneous characters.
        3. Be mindful of how many days of data to pull, as certain queries may specify n days but 
        require more than n to compute the result.
        4. Avoid creating new columns or performing calculations; these will be handled in later steps.
        5. Ensure new column names are self-explanatory and clear for what they are, not what they will be.
        For example do treasury_yield_7_year instead of something like 7y_yield, but don't do
        treasury_price_7_year, even if eventually that column will be transformed into price.
        6. keep is relevant for cols where joins are performed- for example if there is an inner join between
        two dates, these columns are needed to perform the join, but one is irrelevant in the end (one is keep=False)

        Answer ONLY with a JSON in this format:
        {{
  "tables": [
    {{
      "name": "<db_table_name>",
      "alias": "<alias>",
      "columns": [
        {{
          "original_name": "<col_name_in_db>",
          "alias": "<col_alias_in_output>",
          "keep": true // or false
        }}
      ]
    }}
  ],
  "joins": [
    {{
      "left_table_alias": "<alias>",
      "right_table_alias": "<alias>",
      "left_column": "<col_name_in_db>", 
      "right_column": "<col_name_in_db>",
      "join_type": "inner" // or outer, left, right
    }}
  ]
}}
        Constraints:
        - Join types must be one of: "inner", "left", "right", "outer".
        - Join column names must be the OLD name, not the NEW name 
        - Joins will be applied in order of the json. Note the order does matter here. For example 
        if the first join is inner(A,B) then next is outer(B, C), what is really happening is 
        outer(C, inner(A,B)), i.e. not inner(A, outer(B,C))

        Database Schema: {schema_context}

        Example User Query: Calculate the correlation between 7-year treasury yields and stocks' 
        close prices over the last 30 days.

        Example Labeled Answer:
        ```
        json
        {{
    "tables": [
        {{
        "name": "ohlc",
        "alias": "ohlc",
        "columns": [
            {{
            "original_name": "date",
            "alias": "ohlc_date",
            "keep": true
            }},
            {{
            "original_name": "close",
            "alias": "stock_close",
            "keep": true
            }}
        ]
        }},
        {{
        "name": "treasury_yields",
        "alias": "treasury_yields",
        "columns": [
            {{
            "original_name": "date",
            "alias": "tsy_date",
            "keep": false
            }},
            {{
            "original_name": "yield_7_year",
            "alias": "tsy_yield_7_year",
            "keep": true
            }}
        ]
        }}
    ],
    "joins": [
        {{
        "left_table_alias": "ohlc",
        "right_table_alias": "treasury_yields",
        "left_column": "date",
        "right_column": "date",
        "join_type": "inner"
        }}
    ]
    }}
    ```
        Answer for the following user query: {user_query}
        ''' 

        try:
            llm_response = self.config.llm.invoke(prompt).content
        except Exception as e:
            raise RuntimeError(f"LLM invocation for SQL JSON failed: {e}")
        self._log("=" * 50)
        self._log("LLM Response (raw):")
        self._log(llm_response)

        try:
            sql_json_str = extract_query(llm_response, type='json')
            sql_json = json.loads(sql_json_str)
        except Exception as e:
            raise ValueError(f"SQL JSON Extraction Failed (2nd attempt): {e}")

        is_valid, err_msg = self.validate_sql_json(sql_json)
        if not is_valid:
            raise ValueError(f"LLM failed to fix the JSON: {err_msg}")

        return sql_json

    
    def compile_sql(self, sql_json: dict) -> str:
        """
        Transforms the validated JSON (with no filters) into a SQL SELECT statement.

        The JSON must have:
        {
        "tables": [
            {
            "name": "ohlc",
            "alias": "ohlc",
            "columns": [
                {
                "original_name": "date",
                "alias": "ohlc_date",
                "keep": true
                },
                ...
            ]
            },
            ...
        ],
        "joins": [
            {
            "left_table_alias": "ohlc",
            "right_table_alias": "treasury_yields",
            "left_column": "date",
            "right_column": "date",
            "join_type": "inner"
            }
        ]
        }

        Returns:
            A string representing the SQL query (SELECT + FROM + JOIN).
        """
        # 1) Build SELECT clause
        select_parts = []
        table_alias_map = {}
        for tbl in sql_json["tables"]:
            table_alias_map[tbl["alias"]] = tbl["name"]
            for col in tbl["columns"]:
                if col["keep"] is True:
                    original = col["original_name"]
                    alias = col["alias"]
                    select_parts.append(f'"{tbl["alias"]}"."{original}" AS "{alias}"')

        if not select_parts:
            raise ValueError("No columns to keep in the final query.")

        select_clause = ",\n    ".join(select_parts)

        # 2) Build FROM + JOIN
        joins = sql_json.get("joins", [])
        if len(joins) == 0:
            # No joins: just use the first table
            if len(sql_json["tables"]) == 0:
                raise ValueError("No tables available to compile SQL.")
            first_tbl = sql_json["tables"][0]
            from_clause = f'"{first_tbl["name"]}" "{first_tbl["alias"]}"'
        else:
            # Use the first join's left_table_alias as the "base"
            base_alias = joins[0]["left_table_alias"]
            base_table_name = table_alias_map[base_alias]
            from_clause = f'"{base_table_name}" "{base_alias}"'
            for j in joins:
                right_table_name = table_alias_map[j["right_table_alias"]]
                join_type = j["join_type"].upper()
                from_clause += (
                    f'\n{join_type} JOIN "{right_table_name}" "{j["right_table_alias"]}" '
                    f'ON "{j["left_table_alias"]}"."{j["left_column"]}" = '
                    f'"{j["right_table_alias"]}"."{j["right_column"]}"'
                )

        # 3) Construct final SQL
        query = f"""
    SELECT
        {select_clause}
    FROM {from_clause}
    """.strip()

        return query


    def get_python_prompt(self, user_query: str, df: pd.DataFrame) -> str:
        """
        Given the user query and a pandas DataFrame, produce a textual prompt describing
        how to perform the Python computations that answer the query.
        """
        self._log("=" * 50)
        self._log("[STEP 3] Generating Python prompt...")

        prompt = f''' 
        Given a user query and a pandas dataframe with the relevant data, only write 
        CODE DESCRIPTION: CODE DESCRIPTION, where CODE DESCRIPTION is a prompt 
        that (a) gives a high level goal, (b) gives a step by step method that describes how 
        to take the dataframe (called df) and write python code to perform 
        relevant computations to answer the user query. Don't write any code, but write 
        the prompt such that if an independent python master with access to df + your instructions could 
        easily answer the original user query. Be specific about how to perform the computations,
        including any relevant math, what functions to use (assume pandas, numpy access). 

        Example User Query: Calculate the correlation between 7 year treasury yields and stocks close over the last 30 days
        in the table.
        Example Dataframe (df.head()): 
        Date,Treasury Yield (7-Year),Stock Close
        2024-01-01 00:00:00,4.113933,84.676268
        2023-12-29 00:00:00,4.117221,100.393128
        2023-12-28 00:00:00,2.391113,112.97598
        2023-12-27 00:00:00,1.482054,119.224503
        2023-12-26 00:00:00,4.187207,108.335695

        Example Answer:
        CODE DESCRIPTION: 
        Overall Goal: Calculate correlation between treasury_yield_7_year and stock close over the most recent 30 days.
        Step 1: take df with cols treasury_yield_7_year, stock_close, date 
        Step 2: filter df by sorting by date and taking only the most recent 30 days
        Step 3: calculate correlation between treasury_yield_7_year and stock_close using pandas corr function.

        df.iloc[:5,:].T: {df.iloc[:5,:].T}
        User Query: {user_query}
        '''
        try:
            py_prompt = self.config.llm.invoke(prompt).content
        except Exception as e:
            raise RuntimeError(f"LLM invocation for Python prompt failed: {e}")

        if not isinstance(py_prompt, str) or not py_prompt.strip():
            raise ValueError("Python Prompt Generation Failed: Did not receive a valid string from LLM.")
        
        self._log("=" * 50)
        self._log("[STEP 3] Completed: Python prompt generated.")
        self._log("Generated Python Prompt:")
        self._log(py_prompt)

        return py_prompt

    def execute_python(self, py_prompt: str, df: pd.DataFrame):
        """
        Generate Python code based on the py_prompt, execute it, and return the 'result' variable.
        """
        self._log("=" * 50)
        self._log("[STEP 4] Generating and executing Python code...")

        py_code_request = f''' 
        Given a pandas dataframe df and a description to perform a specific computation, 
        generate syntactically correct python code wrapped in python QUERY` that takes 
        the raw dataframe and performs any computations to fully answer the user's query. 
        Assume access to NumPy (v{np.__version__}), Pandas (v{pd.__version__}) and that 
        the dataframe is called df. The output of the code should be the variable that 
        contains the result of the user's query (call this variable result)

        Example Dataframe (df): 
        Date,Treasury Yield (7-Year),Stock Close
        2024-01-01 00:00:00,4.113933,84.676268
        2023-12-29 00:00:00,4.117221,100.393128
        2023-12-28 00:00:00,2.391113,112.97598
        2023-12-27 00:00:00,1.482054,119.224503
        2023-12-26 00:00:00,4.187207,108.335695

        Example Prompt: Given df with cols treasury_yield_7_year, stock_close, date, use pandas corr function
        to compute the correlation between treasury_yield_7_year and stock close. 

        Example Labeled Answer: 
        ```
        python
        ### calculate corr btwn 7yr tsy and stock closes 
        df     = df.sort_values('date')[:30]
        result = df['treasury_yield_7_year'].corr(df['stock_close'])    
        ```
        df.iloc[:5,:].T: {df.iloc[:5,:].T}
        Prompt: {py_prompt}
        '''
        try:
            llm_response = self.config.llm.invoke(py_code_request).content
        except Exception as e:
            raise RuntimeError(f"LLM invocation for Python code generation failed: {e}")

        # Extract python code block
        try:
            code = extract_query(llm_response, type='python')
        except Exception as e:
            raise ValueError(f"Python Code Extraction Failed: {e}")

        self._log("=" * 50)
        self._log("Generated Python Code:")
        self._log(code)

        namespace = {'pd': pd, 'np': np, 'df': df, 'plt': plt}
        try:
            exec(code, namespace)
        except Exception as e:
            raise RuntimeError(f"Python Execution Failed: error executing python: {e}")

        result = namespace.get('result', None)
        self._log("=" * 50)
        self._log("[STEP 4] Result from executed code:")
        self._log(str(result))

        return result

if __name__ == '__main__':
    config     = Config()
    agent      = BasicAgent(config)
    user_queries = ['''For the day with the lowest ratio of 5y tsy yield to 10y tsy yield 
    among days where USD to GBP was greater than 0.75, calculate the percentage difference 
    between the EUR equivalent of the close price and the JPY equivalent of the open price 
    in the ohlc table.''','''On the day where the absolute difference between 7y tsy yield 
    and 10y tsy yield was maximum, compute the ratio of the USD equivalent of the difference 
    between high and low prices in the ohlc table to the product of USD:EUR and USD:JPY 
    for that day.''', '''"For the day where the sum of usd_to_eur, usd_to_gbp, and usd_to_jpy 
    was closest to 150, calculate the weighted average of the EUR equivalent of open, the GBP 
    equivalent of close, and the JPY equivalent of high, with weights being the corresponding 
    treasury yields'''] 

    # user_query = '''For s in [1,2], of the days where the stock price 
    # movement := close - open was more than s std deviations from the mean, look at the distribution 
    # of 7yr tsy yield - 5yr tsy yield. To visualize this, assume access to matplotlib.pyplot as plt and 
    # make 2 plots, the left where s = 1 and a histogram of 7yr - 5yr tsy yields with lines at 25 percentile,
    # 50th percentile, 75th, and then right same with s=2. Then return the median when s=1. 
    # '''
    for user_query in user_queries:
        try:
            result = agent.run(user_query)  
            print("Final numeric result:", result)
        except Exception as e:
            print(e)
        print(f'/' * 50)