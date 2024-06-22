                                                              #Celebrity Search App

import os

from constants import openai_api_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_api_key


## Streamlit App Frame
st.title("🌟Celebrity Search Results")
st.write("Enter the name of the celebrity to search for their details")
input_text = st.text_input("Enter Celebrity Name")



## Memmory
person_memory = ConversationBufferMemory(input_key='name', memory_key='person_history')
dateofbirth_memory = ConversationBufferMemory(input_key='name', memory_key='dob_history')
desc_memory = ConversationBufferMemory(input_key='name', memory_key='desc_history')


## OpenAI LLMs
llm = OpenAI(temperature=0.8)


## Prompt Templates
first_input_prompt = PromptTemplate(
    input_text = ['name'],
    template = "Tell me about celebrity {name}"
)
## Chain initialise
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory) 


## Prompt Template
second_input_prompt = PromptTemplate(
    input_text = ['person'],
    template = "When {person} was born"
)
## Chain initialise
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='DateofBirth', memory=dateofbirth_memory)



## Prompt Template
third_input_prompt = PromptTemplate(
    input_text = ['DateofBirth'],
    template = "Mention 5 major events happened around {DateofBirth} in the world"
)
## Chain initialise
chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=desc_memory)




## Connecting all the chains in Sequence
parent_chain = SequentialChain(
                chains=[chain,chain2, chain3], input_variables=['name'], output_variables=['person', 'DateofBirth','description'], verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))
    
    with st.expander('Person Name'):
        st.info(person_memory.buffer)
        
    with st.expander('Major Events'):
        st.info(desc_memory.buffer)
