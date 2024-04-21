from crewai import Agent, Task, Crew, Process
import os

os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] ='llama3-70b-8192'  # Adjust based on available model
# Please ensure os.environ["OPENAI_API_KEY"] is set here or in start up script to your api key from https://console.groq.com/docs/api-keys

email ="Hi Ez, Mum has asked me to ask you to call.  Try to keep in touch.  I know it is difficult sometimes"

classifier = Agent(
    role = "email classifier",
    goal = "accurately classify emails based on their importance.  Give every email one of these ratings: important, casual, spam.",
    backstory = "You are an AI assistant whose only job is to classify emails accurately and honestly.  Do not be afriad to give emails a bad rating if they are not important. Your job is to help the user manage their inbox.",
    verbose = True,
    allow_delegation = False,
)

responder = Agent(
    role = "email responder",
    goal = "Based on the importance of the email, if the email is rated 'spam' ignore the email, so do not write a response.  If it is 'important' or 'casual' write a concise and simple response.  If the email is rated 'important' write a formal response.  If the email is rated 'casual' write a casual response. And . No matter what be concise. Remember do not write response for 'spam'.  A response is only needed for 'important' or 'casual' rated emails",
    backstory = "You are an AI assistant whose only job is to write short responses to emails based their importance rating received from the 'classifier' agent. Remember The importance (rating) will be provided to you by the 'classifier' agent.",
    verbose = True,
    allow_delegation = False,
)

classify_email = Task(
    description = f"classify the following email: ```{email}```",
    agent = classifier,
    expected_output = "One of these options: 'important', 'casual', or 'spam'"
)

respond_to_email = Task(
    description = f"Respond to the email: ```{email}``` based on the importance (rating) provided by the 'classifier' agent. Note no response is needed for spam emails.",
    agent = responder,
    expected_output = "a very concise response to the email based on the importance provided by the 'classifier' agent. And no email response if the email is classified by the 'classifier' as spam.  Just say '****SPAM*****'"
)

crew = Crew(
    agents = [classifier, responder],
    tasks = [classify_email, respond_to_email],
    verbose = 2,
    process = Process.sequential,
    max_rpm=10,
)

output = crew.kickoff()
print(output)
