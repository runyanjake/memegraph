# memegraph
I previously did meme generation via Langchain in https://github.com/runyanjake/memechain. Here is the same project but written using Langgraph instead, since that seems to have superseded Langchain in recent time.

The main reason to make the switch is to explore [multi-agent](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/multi-agent-collaboration.ipynb) workflows, which hopefully we can apply to this project at some point.

## Setup 

### Python 
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
...
deactivate
```

## Usage 
Can specify prompt when calling script or will use default prompt.
```
python main.py 
python main.py --prompt "Generate an image for the 'Leonardo dicaprio cheers' meme with the text 'LANGGRAPH MEMES'."
```
