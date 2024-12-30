import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent import Agent, Config
from helper import get_paths, extract_query, get_schema_context

class BasicAgent(Agent):

    def __init__(self, config: Config):
        super().__init__(config)
        self.path_map = get_paths()  

    def _log(self, message: str):
        """
        Helper method to handle logging to both console and a log file.
        Appends each message as a new line.
        """
        print(message)
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
        5. Ensure new column names are self-explanatory and clear.

        Answer ONLY with a JSON in this format:
        {{
            "tables": {{
                "table1": [
                    ["original_column_name", "new_column_name"],
                    ...
                ],
                "table2": [...],
            }},
            "joins": [
                ["tableA", "tableB", "tableA_join_column", "tableB_join_column", "join_type"]
            ]
        }}

        Constraints:
        - Join types must be one of: "inner", "left", "right", "outer".
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
            "tables": {{
                "ohlc": [
                    ["date", "date"],
                    ["close", "stock_close"]
                ],
                "treasury_yields": [
                    ["date", "date"],
                    ["yield_7_year", "tsy_yield_7_year"]
                ]
            }},
            "joins": [
                ["ohlc", "treasury_yields", "date", "date", "inner"]
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

        # Extract JSON from the response
        try:
            sql_json_str = extract_query(llm_response, type='json')
            sql_json = json.loads(sql_json_str)
        except json.JSONDecodeError as e:
            raise ValueError("SQL JSON Extraction Failed: LLM response is not valid JSON.")
        except Exception as e:
            raise ValueError(f"SQL JSON Extraction Failed: {e}")

        # Validate join types
        valid_join_types = {"inner", "left", "right", "outer"}
        for join in sql_json.get("joins", []):
            if len(join) != 5 or join[-1] not in valid_join_types:
                raise ValueError(f"SQL JSON Validation Failed: Invalid join type or structure: {join}")

        return sql_json

    def compile_sql(self, sql_json: dict) -> pd.DataFrame:
        """
        Compiles a JSON definition of tables and joins into a SQLite SELECT query,
        executes it, and returns a pandas DataFrame.
        """
        self._log("=" * 50)
        self._log("[STEP 2] Compiling and running SQL...")

        tables = sql_json.get("tables", {})
        joins = sql_json.get("joins", [])

        # 1) Build the projection columns (the SELECT part)
        select_columns = []
        for table_name, column_pairs in tables.items():
            for original, alias in column_pairs:
                select_columns.append(f'"{table_name}"."{original}" AS "{alias}"')

        if not select_columns:
            raise ValueError("No columns provided in 'tables' field of SQL JSON.")

        columns_str = ",\n    ".join(select_columns)

        # 2) Build the FROM/JOIN parts of the query
        if joins:
            base_table = joins[0][0]
            from_clause = f'"{base_table}"'
            for (tbl_left, tbl_right, left_on, right_on, join_type) in joins:
                join_type_upper = join_type.upper() + " JOIN"
                from_clause += (
                    f'\n{join_type_upper} "{tbl_right}" '
                    f'ON "{tbl_left}"."{left_on}" = "{tbl_right}"."{right_on}"'
                )
        else:
            # If no joins, just take the first table
            base_table = list(tables.keys())[0]
            from_clause = f'"{base_table}"'

        # 3) Put it all together
        query = f'''
SELECT
    {columns_str}
FROM {from_clause}
'''.strip()
        self._log("=" * 50)
        self._log("Generated SQL Query:")
        self._log(query)

        try:
            df = pd.read_sql(query, self.config.engine)
        except Exception as e:
            raise RuntimeError(f"SQL Execution Failed: {e}")

        self._log("=" * 50)
        self._log("[STEP 2] SQL Query executed. Here's df.head():")
        self._log(df.head().to_string())

        return df

    def get_python_prompt(self, user_query: str, df: pd.DataFrame) -> str:
        """
        Given the user query and a pandas DataFrame, produce a textual prompt describing
        how to perform the Python computations that answer the query.
        """
        self._log("=" * 50)
        self._log("[STEP 3] Generating Python prompt...")

        prompt = f''' 
        Given a user query and a pandas dataframe with the relevant data, only write 
        CODE DESCRIPTION: CODE DESCRIPTION, where CODE DESCRIPTION is a prompt that 
        describes how to take the dataframe (called df) and write python code to perform 
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
        CODE DESCRIPTION: Given df with cols treasury_yield_7_year, stock_close, date, use pandas corr function
        to compute the correlation between treasury_yield_7_year and stock close over the most recent 30 days.

        df.head(): {df.head()}
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
        df.head(): {df.head()}
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
    user_query = '''For s in [1,2], of the days where the stock price 
    movement := close - open was more than s std deviations from the mean, look at the distribution 
    of 7yr tsy yield - 5yr tsy yield. To visualize this, assume access to matplotlib.pyplot as plt and 
    make 2 plots, the left where s = 1 and a histogram of 7yr - 5yr tsy yields with lines at 25 percentile,
    50th percentile, 75th, and then right same with s=2. Then return the median when s=1. 
    '''
    result = agent.run(user_query)
    print("Final numeric result:", result)
