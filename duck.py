import os
import google.generativeai as genai
from duckduckgo_search import DDGS
from crewai import Agent, Task, Crew, Process

# Configure the Gemini API with your API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

class GeminiWrapper:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
        )
        self.callbacks = []

    def generate(self, prompt):
        chat_session = self.model.start_chat(history=[])

        if isinstance(prompt, tuple):
            prompt = " ".join(str(item) for item in prompt if item is not None)
        elif not isinstance(prompt, str):
            prompt = str(prompt)

        print(f"Sending prompt to Gemini: {prompt}")

        try:
            response = chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def bind(self, stop=None):
        return self.generate

# Create an instance of the Gemini wrapper
gemini_model = GeminiWrapper()

# Function to search for tech news using DuckDuckGo
def search_duckduckgo(query):
    try:
        results = DDGS(query, max_results=5)
        return results
    except Exception as e:
        print(f"Error searching DuckDuckGo: {str(e)}")
        return []

# Create the tech news analyst agent
tech_news_analyst = Agent(
    role='Tech News Analyst',
    goal='Analyze the latest tech news and provide insights on industry trends',
    backstory='You are an experienced tech news analyst with a track record of identifying key trends and insights in the technology sector.',
    verbose=True,
    allow_delegation=False,
    llm=gemini_model,
    tool=search_duckduckgo
)

# Create tasks for the agent
def create_task(description, expected_output):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=tech_news_analyst,
        context=[{
            'description': description,
            'expected_output': expected_output
        }]
    )

def analyze_tech_news():
    tasks = [
        create_task('Analyze the current trends in the tech industry', "Industry trends report"),
        create_task('Research the latest advancements in technology', "Technology advancements report"),
        create_task('Summarize key insights from recent tech news articles', "Tech news insights")
    ]
    
    # Search DuckDuckGo for additional tech news insights
    search_results = search_duckduckgo("latest tech news")
    print(f"DuckDuckGo search results for tech news: {search_results}")

    tech_news_crew = Crew(
        agents=[tech_news_analyst],
        tasks=tasks,
        verbose=2,
        process=Process.sequential
    )
    
    try:
        result = tech_news_crew.kickoff()
        for task in tasks:
            print(f"Output for task '{task.description}': {task.output}")
        return result
    except Exception as e:
        print(f"Error during tech news analysis: {str(e)}")
        return "An error occurred during the tech news analysis process."

# Example usage
if __name__ == "__main__":
    analysis_result = analyze_tech_news()
    print("Tech Industry Analysis:")
    print(analysis_result)