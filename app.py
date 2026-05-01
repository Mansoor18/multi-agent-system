import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ── Page config ──
st.set_page_config(
    page_title="Multi-Agent Support System",
    page_icon="🤖",
    layout="wide"
)

# ── Header ──
st.title("🤖 Multi-Agent Customer Support System")
st.markdown("*Powered by CrewAI — 3 specialised agents collaborate to resolve your query*")
st.divider()

# ── How it works ──
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🔍 Classifier Agent**\nTriages your query and assigns category + priority")
with col2:
    st.info("**📚 Research Agent**\nFinds relevant policies, solutions and escalation needs")
with col3:
    st.info("**✍️ Response Agent**\nCrafts a professional, empathetic customer response")

st.divider()

# ── Sample queries ──
st.subheader("Try a sample query or write your own")

samples = {
    "💰 Billing Issue": "I was charged twice for my order #12345 last week and need an immediate refund!",
    "🔧 Technical Problem": "My internet keeps disconnecting every 30 minutes since yesterday. I work from home and this is urgent.",
    "📦 Shipping Query": "My order was supposed to arrive 3 days ago but I haven't received it yet. Order #98765.",
    "↩️ Returns Request": "I received a damaged product and want to return it for a full refund within the 30-day window.",
}

selected = st.selectbox("Sample queries", ["Write my own..."] + list(samples.keys()))

if selected == "Write my own...":
    query = st.text_area("Enter your support query:", height=100,
                         placeholder="Describe your issue here...")
else:
    query = st.text_area("Enter your support query:", value=samples[selected], height=100)

# ── Run button ──
run_btn = st.button("🚀 Run Multi-Agent System", use_container_width=True,
                    type="primary")

# ── Agent execution ──
if run_btn and query.strip():
    from crewai import Agent, Task, Crew, Process

    LLM = "groq/llama-3.1-8b-instant"

    # ── Agent outputs placeholders ──
    st.divider()
    st.subheader("🔄 Agent Pipeline — Live Execution")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 🔍 Classifier Agent")
        classifier_placeholder = st.empty()
        classifier_placeholder.info("⏳ Waiting...")

    with col_b:
        st.markdown("### 📚 Research Agent")
        research_placeholder = st.empty()
        research_placeholder.info("⏳ Waiting...")

    with col_c:
        st.markdown("### ✍️ Response Agent")
        response_placeholder = st.empty()
        response_placeholder.info("⏳ Waiting...")

    # ── Create agents ──
    classifier = Agent(
        role="Customer Query Classifier",
        goal="Analyse incoming customer queries and classify them into one of: BILLING, TECHNICAL, SHIPPING, RETURNS, GENERAL",
        backstory="You are an expert customer support triage specialist with 10 years of experience.",
        llm=LLM,
        verbose=False
    )

    researcher = Agent(
        role="Support Knowledge Researcher",
        goal="Research and identify the most relevant information, policies, and solutions for the customer issue.",
        backstory="You are a seasoned support knowledge base expert who knows all company policies inside out.",
        llm=LLM,
        verbose=False
    )

    responder = Agent(
        role="Customer Response Specialist",
        goal="Craft a professional, empathetic, and solution-focused response to the customer.",
        backstory="You are an expert customer communication specialist known for turning frustrated customers into loyal ones.",
        llm=LLM,
        verbose=False
    )

    # ── Create tasks ──
    classify_task = Task(
        description=f"""Classify this customer query:

        Query: {query}

        Respond with:
        - Category: [BILLING/TECHNICAL/SHIPPING/RETURNS/GENERAL]
        - Priority: [LOW/MEDIUM/HIGH]
        - Reasoning: [Brief explanation]""",
        expected_output="Category, priority and reasoning",
        agent=classifier
    )

    research_task = Task(
        description=f"""Research solutions for this query:

        Query: {query}

        Provide:
        - Key Issues identified
        - Recommended Solutions (step by step)
        - Escalation needed: YES/NO""",
        expected_output="Issues, solutions and escalation recommendation",
        agent=researcher
    )

    response_task = Task(
        description=f"""Write a professional customer support response for:

        Query: {query}

        Requirements:
        - Start with empathy
        - Provide clear actionable steps
        - Set resolution timeline expectations
        - End with follow-up offer
        - Keep between 100-150 words""",
        expected_output="Complete professional customer response ready to send",
        agent=responder
    )

    # ── Run crew ──
    with st.spinner("Agents are working..."):
        classifier_placeholder.warning("🔄 Classifying query...")

        crew = Crew(
            agents=[classifier, researcher, responder],
            tasks=[classify_task, research_task, response_task],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()

        # Update placeholders
        classifier_placeholder.success(f"✅ Done\n\n{classify_task.output.raw if classify_task.output else ''}")
        research_placeholder.success(f"✅ Done\n\n{research_task.output.raw[:300] if research_task.output else ''}...")
        response_placeholder.success(f"✅ Done")

    # ── Final response ──
    st.divider()
    st.subheader("📨 Final Customer Response")
    st.success(str(result))

    # ── Copy button area ──
    st.text_area("Copy this response:", value=str(result), height=200)

elif run_btn and not query.strip():
    st.warning("Please enter a query first.")

else:
    st.markdown("""
    ### How this works
    This system uses **CrewAI** to orchestrate 3 specialised AI agents:

    | Agent | Role | Output |
    |---|---|---|
    | Classifier | Reads query, assigns category + priority | BILLING / HIGH |
    | Researcher | Finds policies and step-by-step solutions | Structured research |
    | Responder | Crafts empathetic, professional reply | Ready-to-send response |

    **Why multi-agent vs single LLM?**
    - Each agent has a focused, minimal context → less hallucination
    - Sequential pipeline creates built-in quality checks
    - Escalation logic built into Research Agent
    """)