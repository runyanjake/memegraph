from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field

from tools.get_memes import get_memes
from tools.caption_image import caption_image
from tools.download_image import download_image

from typing import TypedDict, Optional

import json
import re 


############################# State Definition ################################
class State(TypedDict):
    question: str
    template_name: Optional[str]
    text: Optional[list[str]]
    template_id: Optional[int]
    caption_result: Optional[dict]
    image_path: Optional[str]

############################# LLM Definition ##################################

llm = OllamaLLM(model="llama3")

class MemeInfo(BaseModel):
    template_name: str = Field(..., description="The meme template name")
    text: list[str] = Field(..., description="List of captions/texts for the meme")

parser = PydanticOutputParser(pydantic_object=MemeInfo)

def parse_input(state: State) -> State:
    prompt = f"""
    Extract the meme name and list of texts from the user's request. 
    Return the output in this JSON format:

    {parser.get_format_instructions()}

    User input: {state["question"]}
    """

    response = llm.invoke(prompt)
    parsed = parser.invoke(response)
    print(parsed)
    return {**state, **parsed.dict()}

######################## Graph Components Definition ##########################
def node_get_memes(state: State) -> State:
    memes_str = get_memes()
    memes = memes_str.strip().splitlines()

    for meme in memes:
        match = re.match(r"ID: (\d+), Name: (.+)", meme)
        if match:
            meme_id = int(match.group(1))
            meme_name = match.group(2).strip()

            if meme_name.lower() == state["template_name"].lower():
                print("Found Meme: " + meme)
                return {**state, "template_id": meme_id}

    raise ValueError(f"No matching meme found for template_name: {state['template_name']}")

def node_caption_image(state: State) -> State:
    input_data = {
        "template_id": int(state["template_id"]),
        "text": state["text"]
    }
    json_input = json.dumps(input_data)
    result = caption_image(json_input)
    print(result)
    return {**state, "caption_result": result}

def node_download_image(state: State) -> State:
    caption_result = state["caption_result"]
    print("Generated Image: " + caption_result)
    if isinstance(caption_result, str) and caption_result.startswith("http"):
        download_payload = json.dumps({"url": caption_result})
        result = download_image(download_payload)
        print("Saved to: " + result)
        return {**state, "image_path": result}
    else:
        print("Failed to download image. " + caption_result)

############################## Graph Definition ###############################
graph = StateGraph(State)

graph.add_node("parse_input", parse_input)
graph.add_node("get_memes", node_get_memes)
graph.add_node("caption_image", node_caption_image)
graph.add_node("download_image", node_download_image)

graph.set_entry_point("parse_input")
graph.add_edge("parse_input", "get_memes")
graph.add_edge("get_memes", "caption_image")
graph.add_edge("caption_image", "download_image")
graph.add_edge("download_image", END)

app = graph.compile()

#################################### Run ######################################
question = "Generate an image for the 'two buttons' meme with first text 'generated meme' and second text 'langgraph error'."

final_state = app.invoke({"question": question})
print(final_state["image_path"])
