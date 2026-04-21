from crewai import Crew, Process
from agents import create_classifier_agent, create_research_agent, create_response_agent
from tasks import create_classification_task, create_research_task, create_response_task

def run_support_crew(query):
    print(f"\n{'='*60}")
    print(f"Processing Query: {query}")
    print(f"{'='*60}\n")

    # Create agents
    classifier = create_classifier_agent()
    researcher = create_research_agent()
    responder = create_response_agent()

    # Create tasks
    classify_task = create_classification_task(classifier, query)
    research_task = create_research_task(researcher, query)
    response_task = create_response_task(responder, query)

    # Assemble crew
    crew = Crew(
        agents=[classifier, researcher, responder],
        tasks=[classify_task, research_task, response_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    
    print(f"\n{'='*60}")
    print("FINAL CUSTOMER RESPONSE:")
    print(f"{'='*60}")
    print(result)
    
    return result