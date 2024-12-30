import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent import Agent, Config
from helper import * 

class BasicAgent(Agent):

    def run(self, user_query: str) -> float:
        """
        Takes a user query string, runs it through the get_sql_json3 -> compile_sql -> get_python_prompt3 -> execute_python3 pipeline,
        and returns a float result.
        """
        print('=' * 30)
        print(f'User Query:\n {user_query}')
        start = time.time()

        # 1) Convert the user query to SQL JSON
        sql_json = self.get_sql_json3(user_query)
        # 2) Compile SQL and fetch data into a dataframe
        df = self.compile_sql(sql_json)
        # 3) Generate a Python prompt describing the next steps
        py_prompt = self.get_python_prompt3(user_query, df)
        # 4) Execute the described Python steps and retrieve the result
        result = self.execute_python3(py_prompt, df)

        end = time.time()
        print('=' * 30)
        print(f'Runtime: {round(end - start, 2)} seconds')

        if isinstance(result, (int, float)):
            return float(result)
        else:
            raise ValueError("Result is not a numeric value.")

    def get_sql_json3(self, user_query):
        """Get SQL query as JSON structure"""
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
        ```json
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

        Answer for the following user query: {user_query}
        ''' 
        llm_response = self.config.llm.invoke(prompt).content
        print('=' * 30)
        print(f'LLM Response: {llm_response}')
        sql_json = extract_query(llm_response, type='json')
        try:
            sql_json = json.loads(sql_json)
            print(f'=' * 30)
            print(f'SQL PARSED JSON: {sql_json}')
            valid_join_types = {"inner", "left", "right", "outer"}
            for join in sql_json.get("joins", []):
                if len(join) != 5 or join[-1] not in valid_join_types:
                    raise ValueError(f"Invalid join type or structure: {join}")

            return sql_json
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")
        except Exception as e:
            raise ValueError(f"Error validating SQL JSON: {e}")

    def compile_sql(self, sql_json: dict) -> str:
        """
        Compiles a JSON definition of tables and joins into a SQLite SELECT query.
        """
        tables = sql_json.get("tables", {})
        joins = sql_json.get("joins", [])

        # 1) Build the projection columns (the SELECT part)
        select_columns = []
        for table_name, column_pairs in tables.items():
            for original, alias in column_pairs:
                select_columns.append(f'"{table_name}"."{original}" AS "{alias}"')

        # If no tables at all, we can't form a valid query
        if not tables:
            raise ValueError("No tables were provided. At least one table is required.")

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
            base_table = list(tables.keys())[0]
            from_clause = f'"{base_table}"'

        # 3) Put it all together
        query = f'''
    SELECT
        {columns_str}
    FROM {from_clause}'''.strip()
        print(f'=' * 30)
        print(f'SQL Query: {query}')
        df = pd.read_sql(query, self.config.engine)
        print(f'Query Results:\n{df.to_string()}')
        return df

    def get_python_prompt3(self, user_query, df):
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
        py_prompt = self.config.llm.invoke(prompt).content
        print(f'Generated Python Prompt: {py_prompt}')
        return py_prompt

    def execute_python3(self, py_prompt, df):
        """generate and execute python"""
        py_code = f''' 
        Given a pandas dataframe df and a description to perform a specific computation, 
        generate syntactically correct python code wrapped in ```python QUERY`` that takes 
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
        ``` python
        ### calculate corr btwn 7yr tsy and stock closes 
        df     = df.sort_values('date')[:30]
        result = df['treasury_yield_7_year'].corr(df['stock_close'])
        ```
        df.head(): {df.head()}
        Prompt: {py_prompt}
        '''
        code = extract_query(self.config.llm.invoke(py_code).content, type='python')
        print(f'Python Code:\n {code}')
        
        namespace = {'pd': pd, 'np': np, 'df': df}
        exec(code, namespace)
        result = namespace.get('result', "No result variable found")
        print(f'Result:\n{result}')
        return result

if __name__ == '__main__':
    config     = Config()
    agent      = BasicAgent(config)
    user_query = '''For s in [1,2], of the days where the stock price 
    movement := close - open was more than s std deviations from the mean, look at the distribution 
    of 7yr tsy yield - 5yr tsy yield. To visualize this, assume access to matplotlib.pyplot as plt and 
    make a 2 plots, the left where s = 1 and a histogram of 7yr - 5yr tsy yields with lines at 25 percentile,
    50th percentile, 75th, and then right same with s=2. Then return the median when s=1. 
    '''
    agent.run(user_query)