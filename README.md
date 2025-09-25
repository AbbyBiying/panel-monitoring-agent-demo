## Introduction
# Panel Monitoring Agent

> As a skilled and dedicated member of our quality assurance team, you’re focused on supporting the integrity of our online consumer survey panel.  You diligently monitor key user behaviors on the individual panelist level from characteristics of each new panel signup to every existing member’s daily actions on our platform.  With incredible AI super powers, you can consider each and every user event, rapidly responding to signals of fraud or abuse, asking for Human in the Loop feedback when relevant, and taking meaningful actions to lock down fraudulent accounts and report your findings to our panel administrators.  By reviewing panelist behaviors in aggregate each day, you can identify widespread fraudulent activity or cyber attacks, improving our security posture.
> 

1. Evaluate a new user event and classify relevant signals
    1. Example User Event:
        
        > NEW SIGNUP: It’s 12:44am in my timezone and I just signed up to Forthright.  My email address is [xyz@example.com](mailto:xyz@example.com).  I signed up through a [source name] [campaign name] recruitment link.  I’m a 22 year-old female living in New York City.  I work full-time as a Director and make $200,000 per year.”  —> “Is this a suspicious signup?  Have we seen this one before?”
        > 
2. Match to aggregated signal statistics for widespread anomaly detection (e.g., consider “This is one of 200 such instances of this event today. Do we think this event is connected with other events on the platform today?  Let’s review those events.”)
3. Decide if action is needed on the account
    1. (option) Hold Member Account
    2. (option) Remove Member Account
4. Decide if Human in the Loop feedback is needed
5. Decide if a notification is appropriate
6. Log user event and signals detected for aggregated statistics and future comparisons
    1. Log the event
    2. Re-summarize the recent logged events: “Normal daily activity.” vs. “We’re seeing a slight uptick in redemptions to Bitcoint today.  Let’s keep an eye on it.” vs. “Today we’re experiencing a rapid increase in fraudulent signups from Facebook.”
7. [fine tuning orchestration?]

## Setup

### Python version

To get the most out of this course, please ensure you're using Python 3.11 or later. 
This version is required for optimal compatibility with LangGraph. If you're on an older version, 
upgrading will ensure everything runs smoothly.
```
python3 --version
```

### Create an environment and install dependencies
```
$ pip install -r requirements.txt
```

### Running notebooks
If you don't have Jupyter set up, follow installation instructions [here](https://jupyter.org/install).
```
$ jupyter notebook
```

### Setting up env variables
Briefly going over how to set up environment variables. You can also 
use a `.env` file


### Sign up and Set LangSmith API
* Sign up for LangSmith [here](https://smith.langchain.com/), find out more about LangSmith
* and how to use it within your workflow [here](https://www.langchain.com/langsmith), and relevant library [docs](https://docs.smith.langchain.com/)!
*  Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING_V2=true` `LANGSMITH_PROJECT="langchain-academy"`in your environment 

### Set up Tavily API for web search

* Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. 
* You can sign up for an API key [here](https://tavily.com/). 
It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily. 

* Set `TAVILY_API_KEY` in your environment.

### Set up LangGraph Studio

* LangGraph Studio is a custom IDE for viewing and testing agents.
* Studio can be run locally and opened in your browser on Mac, Windows, and Linux.
* See documentation [here](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/#local-development-server) on the local Studio development server and [here](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server). 
* Graphs for LangGraph Studio are in the `module-x/studio/` folders.
* To start the local development server, run the following command in your terminal in the `/studio` directory each module:

```
langgraph dev
```

You should see the following output:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

* To use Studio, you will need to create a .env file with the relevant API keys
* Run this from the command line to create these files for module 1 to 5, as an example:
```
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```
# panel-agent
