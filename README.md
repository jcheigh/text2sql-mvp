# text2sql-mvp
Basic text2sql

### Instructions:
- Store OPENAI_API_KEY={key} in data/keys.env
- Add LangSmith env variables for better monitoring 
    - LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
- Run python deploy.py, then basic_agent.py 
### Subgoal 1- Max performance on small dataset, clear questions
- Timeline: January 6th
- Change SQL JSON form
- Refactor with LangGraph, Structured Outputs
- Python Step 1- Data Exploration (cycle where LLM can do stuff like df.head() on its own)
    - Must limit cycle, limit output
- Python Step 2- Feature Engineering (filter/add columns)
- Python Step 3- Analytics (e.g. vol calculations )
- Python Step 4- Visualization (e.g. visualization)
    - For now basic but later this will be a different agent likely
- Validator/Confidence Level
- Improve Output, i.e. ensuring which variable it is saved in
- Final Milestone: can’t come up with clear question on data in small database that it gets wrong.
