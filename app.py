import streamlit as st
from openagi.actions.tools.webloader import WebBaseContextTool
from openagi.actions.tools import DuckDuckGoSearch
from openagi.actions.files import WriteFileAction, ReadFileAction
from openagi.agent import Admin
from openagi.llms.azure import AzureChatOpenAIModel
from openagi.memory import Memory
from openagi.planner.task_decomposer import TaskPlanner
from openagi.worker import Worker
from dotenv import load_dotenv

load_dotenv()
import os



# Load the configuration and initialize the AzureChatOpenAIModel
config = AzureChatOpenAIModel.load_from_env_config()
llm = AzureChatOpenAIModel(config=config)

# Define the workers
researcher = Worker(
    role="Research Mathematican",
    instructions="""You are given the Math question, find the answer using search results and return the correct answer to the question. If answer is not present, then generate `Answer not found on web`
    You must generate 100% results, as the application is for students. You must generate both Question and Answer as the response""",
    actions=[DuckDuckGoSearch, WebBaseContextTool, WriteFileAction],
)

writer = Worker(
    role="Math Professor",
    instructions="""
    You are an expert Math professor, who helps student learn Math in step-by-step solution  \
    You are provided with context that contains both Question and Answer \
    Based on the given Question and Answer generate an accurate step-by-step solution within 3-4 meaningful steps that makes sense \
    Remember being an expert you never lie, so be truthful \
    """,
    actions=[ReadFileAction, DuckDuckGoSearch, WebBaseContextTool, WriteFileAction],
)

# Define the admin
admin = Admin(
    actions=[DuckDuckGoSearch],
    planner=TaskPlanner(human_intervene=False),
    memory=Memory(),
    llm=llm,
)
admin.assign_workers([researcher, writer])

# Create the Streamlit interface
st.title("Math Question Solver")

# Input box for user to enter the query
query = st.text_input("Enter your math question:")

# Button to run the query
if st.button("Solve"):
    if query:
        # Run the admin with the given query
        res = admin.run(
            query=query,
            description="Search and solve the given question",
        )
        # Display the result
        st.write("Result:")
        st.write(res)
    else:
        st.write("Please enter a query.")
