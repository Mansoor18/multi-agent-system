from crew import run_support_crew

if __name__ == "__main__":
    # Test queries
    queries = [
        "I was charged twice for my order #12345 last week and I need a refund immediately!",
        "My internet keeps disconnecting every 30 minutes since yesterday.",
    ]
    
    for query in queries:
        result = run_support_crew(query)
        print("\n" + "="*60 + "\n")