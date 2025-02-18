import os
import re
import chainlit as cl
import pandas as pd

from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.legacy import GraphRAGExtractor, GraphRAGQueryEngine

from llama_index.graph_stores.neo4j import GraphRAGStore

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the LLM (using GPT-4 in this example)
llm = OpenAI(model="03-mini")

##########################
# Step 1: Load Data
##########################
def load_news_data():
    url = "https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/news_articles.csv"
    news = pd.read_csv(url)[:50]
    documents = [
        Document(text=f"{row['title']}: {row['text']}")
        for i, row in news.iterrows()
    ]
    return documents

##########################
# Step 2: Split Documents
##########################
def split_documents(documents):
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Total nodes created: {len(nodes)}")
    return nodes

##########################
# Step 3: Define KG Extraction Prompt & Parser
##########################
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract:
   - entity_name: Name of the entity (capitalized)
   - entity_type: Type of the entity
   - entity_description: A brief description of the entity's attributes and activities
   Format each entity as ("entity"$$$$"<entity_name>"$$$$"<entity_type>"$$$$"<entity_description>")

2. From the entities identified in step 1, identify all pairs of related entities.
   For each related pair, extract:
   - source_entity: name of the source entity
   - target_entity: name of the target entity
   - relation: relationship between the source and target entity
   - relationship_description: explanation for why these entities are related
   Format each relationship as ("relationship"$$$$"<source_entity>"$$$$"<target_entity>"$$$$"<relation>"$$$$"<relationship_description>")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""

# Patterns for extracting entities and relationships from the LLM output
entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

def parse_fn(response_str: str):
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships

##########################
# Step 4: Build GraphRAGExtractor
##########################
kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    parse_fn=parse_fn,
    max_paths_per_chunk=2,
)

##########################
# Step 5: Setup GraphRAGStore (using Neo4j)
##########################
graph_store = GraphRAGStore(
    username="neo4j",
    password="neo4j",  # Replace with your Neo4j password
    url="bolt://localhost:7687"
)

##########################
# Step 6: Build the Property Graph Index
##########################
def build_property_graph_index(nodes):
    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )
    # Build communities and generate summaries
    index.property_graph_store.build_communities()
    return index

##########################
# Step 7: Build the GraphRAG Query Engine
##########################
def build_query_engine(index):
    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )
    return query_engine

##########################
# Full Pipeline
##########################
def build_graph_rag_pipeline():
    documents = load_news_data()
    nodes = split_documents(documents)
    index = build_property_graph_index(nodes)
    query_engine = build_query_engine(index)
    return query_engine

# Build the GraphRAGQueryEngine on startup
query_engine = build_graph_rag_pipeline()

##########################
# Chainlit Chat Interface
##########################
@cl.on_message
def main(message: str):
    try:
        # Process the user query with the GraphRAG query engine
        response = query_engine.query(message)
        cl.send_message(response.response)
    except Exception as e:
        cl.send_message(f"Error: {e}")
