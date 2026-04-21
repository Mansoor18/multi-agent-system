from crewai import Task

def create_classification_task(agent, query):
    return Task(
        description=f"""Classify the following customer query into 
        exactly one category (BILLING, TECHNICAL, SHIPPING, RETURNS, GENERAL).
        
        Customer Query: {query}
        
        Respond with:
        - Category: [CATEGORY]
        - Reasoning: [Brief explanation]
        - Priority: [LOW/MEDIUM/HIGH]""",
        expected_output="Category, reasoning, and priority level for the query",
        agent=agent
    )

def create_research_task(agent, query):
    return Task(
        description=f"""Based on this customer query, identify the key 
        issues and relevant solutions:
        
        Customer Query: {query}
        
        Provide:
        - Key Issues: [List the main problems]
        - Relevant Policies: [Any applicable policies]
        - Recommended Solutions: [Step by step solutions]
        - Escalation needed: [YES/NO and why]""",
        expected_output="Detailed research with issues, policies, solutions and escalation recommendation",
        agent=agent
    )

def create_response_task(agent, query):
    return Task(
        description=f"""Create a professional customer support response 
        for this query using the research and classification provided 
        by your team members:
        
        Original Customer Query: {query}
        
        Your response must:
        - Start with empathy acknowledgement
        - Provide clear actionable steps
        - Set expectations on resolution time
        - End with a follow up offer
        - Be between 100-150 words""",
        expected_output="A complete professional customer support response ready to send",
        agent=agent
    )