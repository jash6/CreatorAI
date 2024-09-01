# RAG only Approach

### Imports
import os
import shutil
import time
from typing import List
from tqdm import tqdm

import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI

import openai
import textwrap

### Constants/ Should go in env file
CHANNEL_URL = "Enter Channel URL"
CHROMA_PATH = "/content/chroma/{channel_id}"
OPENAI_API_KEY = "Enter OpenAI API Key"

### Fetching Data and generating Transcripts

def get_video_ids(channel_url: str) -> List[str]:
    videos = scrapetube.get_channel(channel_url=channel_url)
    return [video['videoId'] for video in videos]

def fetch_transcripts(video_ids: List[str]) -> List[str]:
    transcripts = []
    for video_id in tqdm(video_ids, desc="Fetching transcripts"):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            script = ' '.join(entry['text'] for entry in transcript)
            transcripts.append(script)
        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video {video_id}. Skipping.")
        except NoTranscriptFound:
            print(f"No transcript found for video {video_id}. Skipping.")
        except Exception as e:
            print(f"An error occurred with video {video_id}: {str(e)}. Skipping.")
    return transcripts
video_ids = get_video_ids(CHANNEL_URL)
transcripts = fetch_transcripts(video_ids)

### Splitting the fetched Data and generating Embeddings to store

def split_text(transcripts: List[str]) -> List[Document]:
    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )

    documents = [Document(page_content=transcript) for transcript in transcripts]
    chunks = text_splitter.split_documents(documents)

    unique_chunks = list({chunk.page_content: chunk for chunk in chunks}.values())

    max_chunks = 1000
    if len(unique_chunks) > max_chunks:
        print(f"Limiting to {max_chunks} chunks to avoid rate limits.")
        unique_chunks = unique_chunks[:max_chunks]

    print(f"Split {len(documents)} documents into {len(unique_chunks)} unique chunks.")

    if unique_chunks:
        print("Example chunk:")
        print(unique_chunks[0].page_content)
        print(unique_chunks[0].metadata)

    return unique_chunks

def save_to_chroma(chunks: List[Document], batch_size: int = 100, max_retries: int = 5):
    """Save document chunks to Chroma vector store with rate limit handling."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
        batch = chunks[i:i+batch_size]
        retry_count = 0
        while retry_count < max_retries:
            try:
                db.add_documents(batch)
                break
            except openai.RateLimitError as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed to process batch after {max_retries} retries. Error: {str(e)}")
                    break
                wait_time = 2 ** retry_count
                print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
    """Generate vector database in Chroma from YouTube transcripts."""
    chunks = split_text(transcripts)
    save_to_chroma(chunks)

if __name__ == "__main__":
    generate_data_store()

### Query the entire RAG System

def query_rag(query_text):
    embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    context_text = results[0][0].page_content
    return context_text

### Chat like conversation with RAG

##### Without taking into consideration the personality of the user
llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature= 0.8)

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    You have the same knowledge and personality as the person who has the following context.
    Answer the question as if the person with the context is interacting.
    Use the additional information provided to enhance your responses.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("system", "Additional information from RAG: {rag_content}"),
    ("human", "{input}"),
])

prompt_template = """Write a 200 character summary of the following:
"{context}"
CONCISE SUMMARY:"""
prompt2 = PromptTemplate.from_template(prompt_template)

chain = prompt | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

def run_conversation():
    context = [{"role": "assistant", "content": "Hi there. I am {Enter Creator Name}. What do you want to discuss today with me?"}]
    print(f"\n{"Enter Creator Name"}: {context[0]['content']}")
    while True:
        user_input = input("You: ")
        rag_content = query_rag(user_input)
        if len(rag_content) > 200:
          rag_content = chain2.invoke({"context":rag_content})
        response = chain.invoke({
            "history": context,
            "input": user_input,
            "rag_content": rag_content
        })
        wrapped_response = textwrap.fill(response, width=80)

        print(f"\n{"Enter Creator Name"}:\n{wrapped_response}")
        context.append({"role": "human", "content": user_input})
        context.append({"role": "assistant", "content": response})
run_conversation()

text_combined = ' '.join(transcripts[:20])
len(text_combined)
test = chain2.invoke({"context":text_combined})
splitted_text = split_text(text_combined)
doc = Document(page_content=text_combined)
docs = [doc]

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
print(f"docs = {docs}")

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature= 0)
chain = load_summarize_chain(llm, chain_type="stuff")

result = chain.invoke(docs)

print(result["output_text"])

result["output_text"]

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

prompt_template = """
You are summarizing the communication style of {"Enter Creator Name"}. Based on the following summaries of his latest videos, provide a detailed summary that captures his speaking style, common phrases, interests, and any notable characteristics.
Summaries:
"{text}"
Summary:
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature= 0)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# docs = loader.load()
results = stuff_chain.invoke(docs)["output_text"]

results

{"Enter Creator Name"} = '''
Enter the perosnlaity of the creator.(results in our case)
'''

llm = ChatOpenAI(model="gpt-4", openai_api_key=OPENAI_API_KEY, temperature= 0.8)

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
    You have the same knowledge and personality as the person who has the following context.
    Answer the question as if the person with the context is interacting.
    More information about the creator's tone and the way they talk : {{"Enter Creator Name"}}
    Use the additional information provided to enhance your responses.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("system", "Additional information from RAG: {rag_content}"),
    ("human", "{input}"),
])

prompt_template = """Write a 200 character summary of the following:
"{context}"
CONCISE SUMMARY:"""
prompt2 = PromptTemplate.from_template(prompt_template)

chain = prompt | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

def run_conversation():
    context = [{"role": "assistant", "content": "Hi there. I am {Enter Creator Name}. What do you want to discuss today with me?"}]
    print(f"\n{"Enter Creator Name"}: {context[0]['content']}")
    while True:
        user_input = input("You: ")
        rag_content = query_rag(user_input)
        if len(rag_content) > 200:
          rag_content = chain2.invoke({"context":rag_content})
        response = chain.invoke({
            "history": context,
            "input": user_input,
            "rag_content": rag_content
        })
        wrapped_response = textwrap.fill(response, width=80)

        print(f"\n{"Enter Creator Name"}:\n{wrapped_response}")
        context.append({"role": "human", "content": user_input})
        context.append({"role": "assistant", "content": response})
run_conversation()