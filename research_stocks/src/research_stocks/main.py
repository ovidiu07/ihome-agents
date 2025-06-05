import sys, logging
from crew import StockAnalysisCrew
from tools.market_data_tools import SlackPosterTool

def run():
    inputs = {
        'query': 'What is the company you want to analyze?',
        'company_stock': 'NVDA',
    }
    return StockAnalysisCrew().crew().kickoff(inputs=inputs)


def send_alert_to_slack(channel: str, text: str):
    SlackPosterTool()._run(channel, text)


def generate_fallback_report():
    return "Crew execution failed. No data available."


def safe_run():
    try:
        return run()
    except Exception as e:
        logging.error(f"Critical error in crew execution: {e}")
        send_alert_to_slack("#market-ops", f"Crew execution failed: {e}")
        return generate_fallback_report()

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'query': 'What is last years revenue',
        'company_stock': 'NVDA',
    }
    try:
        StockAnalysisCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

if __name__ == "__main__":
    print("## Welcome to Stock Analysis Crew")
    print('-------------------------------')
    result = safe_run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)
