import gradio as gr
from retrival import LoadAndRetrieve
from rag import rag_pipeline

#Initialize the retrieval class
retriever = LoadAndRetrieve()
#Create emdedding and store in db, if not already done
collection = retriever.create_embeddings()
#Initialize the rag pipeline
rag = rag_pipeline()

def get_answer(query):
    context = retriever.query(collection=collection, query=query)
    response = rag.ask(query=query, context=context)
    return response

demo = gr.Interface(
    get_answer,
    gr.Textbox(label="Ask me Anything!"), 
    gr.JSON(label="Answer"),
    description="A question answering bot using rag. Upload any pdf and question it.",
    title="RAG based Question Answering Bot"
)
demo.launch()