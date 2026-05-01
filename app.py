import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load API key
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page config ──
st.set_page_config(
    page_title="Multi-Agent Support System",
    page_icon="🤖",
    layout="wide"
)

# ── LLM ──
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )

# ── Agent functions ──
def run_classifier(llm, query):
    messages = [
        SystemMessage(content="""You are an expert customer support triage specialist. 
        Classify the query into exactly one category and assign priority.
        Always respond in this exact format:
        Category: [BILLING/TECHNICAL/SHIPPING/RETURNS/GENERAL]
        Priority: [LOW/MEDIUM/HIGH]
        Reasoning: [One sentence explanation]"""),
        HumanMessage(content=f"Classify this customer query: {query}")
    ]
    return llm.invoke(messages).content

def run_researcher(llm, query, classification):
    messages = [
        SystemMessage(content="""You are a seasoned support knowledge base expert. 
        Based on the query and its classification, identify key issues and solutions.
        Always respond with:
        Key Issues: [bullet points]
        Recommended Solutions: [numbered steps]
        Escalation Needed: [YES/NO with reason]"""),
        HumanMessage(content=f"""Query: {query}
        Classification: {classification}
        Research the best solutions for this issue.""")
    ]
    return llm.invoke(messages).content

def run_responder(llm, query, research):
    messages = [
        SystemMessage(content="""You are an expert customer communication specialist.
        Write a professional, empathetic customer support response.
        Requirements:
        - Start with empathy and acknowledgement
        - Provide clear actionable steps
        - Set resolution timeline expectations  
        - End with a follow-up offer
        - Keep between 100-150 words
        - Be warm, professional and solution-focused"""),
        HumanMessage(content=f"""Original Query: {query}
        Research and Solutions: {research}
        Write the final customer response.""")
    ]
    return llm.invoke(messages).content

# ── Header ──
st.title("🤖 Multi-Agent Customer Support System")
st.markdown("*3 specialised AI agents collaborate to resolve your query*")
st.divider()

# ── Agent cards ──
col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🔍 Classifier Agent**\nTriages query → assigns category + priority")
with col2:
    st.info("**📚 Research Agent**\nFinds policies, solutions + escalation needs")
with col3:
    st.info("**✍️ Response Agent**\nCrafts professional empathetic reply")

st.divider()

# ── Sample queries ──
samples = {
    "💰 Billing Issue": "I was charged twice for my order #12345 last week and need an immediate refund!",
    "🔧 Technical Problem": "My internet keeps disconnecting every 30 minutes since yesterday. I work from home and this is urgent.",
    "📦 Shipping Query": "My order was supposed to arrive 3 days ago but I haven't received it yet. Order #98765.",
    "↩️ Returns Request": "I received a damaged product and want to return it for a full refund within the 30-day window.",
}

st.subheader("Try a sample query or write your own")
selected = st.selectbox("Sample queries", ["Write my own..."] + list(samples.keys()))

if selected == "Write my own...":
    query = st.text_area("Enter your support query:", height=100,
                         placeholder="Describe your issue here...")
else:
    query = st.text_area("Enter your support query:",
                         value=samples[selected], height=100)

run_btn = st.button("🚀 Run Multi-Agent System",
                    use_container_width=True, type="primary")

# ── Execution ──
if run_btn and query.strip():
    llm = get_llm()
    st.divider()
    st.subheader("🔄 Agent Pipeline — Live Execution")

    col_a, col_b, col_c = st.columns(3)

    # Classifier
    with col_a:
        st.markdown("### 🔍 Classifier Agent")
        with st.spinner("Classifying..."):
            classification = run_classifier(llm, query)
        st.success(classification)

    # Researcher
    with col_b:
        st.markdown("### 📚 Research Agent")
        with st.spinner("Researching..."):
            research = run_researcher(llm, query, classification)
        st.success(research)

    # Responder
    with col_c:
        st.markdown("### ✍️ Response Agent")
        with st.spinner("Crafting response..."):
            response = run_responder(llm, query, research)
        st.success(response)

    # Final response
    st.divider()
    st.subheader("📨 Final Customer Response")
    st.success(response)
    st.text_area("Copy this response:", value=response, height=200)

elif run_btn and not query.strip():
    st.warning("Please enter a query first.")

else:
    st.markdown("""
    ### How this works
    This system simulates a **Multi-Agent AI Pipeline** using 3 specialised LLM calls:

    | Agent | Role | Output |
    |---|---|---|
    | 🔍 Classifier | Triages query, assigns category + priority | BILLING / HIGH |
    | 📚 Researcher | Finds policies and step-by-step solutions | Structured research |
    | ✍️ Responder | Crafts empathetic professional reply | Ready-to-send response |

    **Why multi-agent vs single LLM?**
    - Each agent has focused, minimal context → less hallucination
    - Sequential pipeline creates built-in quality checks
    - Researcher output feeds Responder — grounded, policy-based replies
    - Escalation logic built into Research Agent
    """)