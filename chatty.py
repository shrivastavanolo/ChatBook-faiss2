import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from langchain.chains import RetrievalQA
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain import OpenAI, LLMChain
import os

os.environ["OPENAI_API_KEY"]="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"

with st.sidebar:
    st.title("ChattyPDF")
    st.markdown('''
                About:
                An LLM powered chatbot built using:
                - Streamlit
                - LangChain
                - OpenAI''')
    add_vertical_space(5)
    st.write("Made by [Shreya Shrivastava](https://www.linkedin.com/in/shreya-shrivastava-b39911244/)")

load_dotenv()
def main():
    llm = OpenAI(temperature=0)
    st.header("Learn your PDF without reading all of it")

    pdf=st.file_uploader("Upload your PDF",type="pdf")

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        store_name=pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                Vectorstore = pickle.load(f)
            # st.write("Embeddings Loaded from Disk")
        else:
            embeddings=OpenAIEmbeddings()
            Vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(Vectorstore,f)
            # st.write("Embeddings Computation Completed")


        ruff = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=Vectorstore.as_retriever())
        
        tools = [
        Tool(
        name="PDF System",
        func=ruff.run,
        description="useful for when you need to answer questions about user given PDF file. Input should be a fully formed question.",
        ),]

        prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )

        message_history = RedisChatMessageHistory(
        url="redis://default:zyRNg3pQk44tfbpw4fQauy1lsuacbDdA@redis-19775.c10.us-east-1-4.ec2.cloud.redislabs.com:19775", ttl=600, session_id=get_script_run_ctx().session_id
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=message_history
        )
        
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        query=st.text_input("Ask question about your file: ")

        if query:
            res=agent_chain.run(input=query)
            st.write(res)
if __name__=='__main__':
    main()