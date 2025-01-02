from flask import Flask, request, render_template
import os

from basic_agent import BasicAgent, Config
from helper import get_schema_context  # Make sure this import is correct

app = Flask(__name__)

# Create a single agent instance for the entire app
agent = BasicAgent(Config(), log_data=False)

@app.route('/', methods=['GET'])
def index():
    """
    Renders the home page: 
    1) Title: Text2SQL Agent
    2) Explanation of what Text2SQL is
    3) Text box for the user query
    4) Database schema and sample data
    """
    schema_context = get_schema_context(agent.config, tbls_to_exclude=[])
    return render_template('index.html', schema_context=schema_context)

@app.route('/run_agent', methods=['POST'])
def run_agent():
    """
    Receives the user query from the form and executes the BasicAgent.
    Returns the agent's log trace and final numeric result to the user.
    """
    user_query = request.form.get('user_query', '')

    # We'll capture logs in memory by overriding the agent._log method
    logs = []
    def custom_log(message: str):
        logs.append(message)
        print(message)  # still print to console if desired

    # Temporarily override _log with our custom logger
    original_log_method = agent._log
    agent._log = custom_log

    result = None
    try:
        result = agent.run(user_query)
        logs.append(f"Final numeric result: {result}")
    except Exception as e:
        logs.append(f"[ERROR] {str(e)}")
    finally:
        # Restore the agent's original _log method
        agent._log = original_log_method

    # Render results template, which displays logs line by line
    return render_template('results.html', logs=logs, result=result)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
