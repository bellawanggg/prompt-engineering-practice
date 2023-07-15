#pip install openai
#pip install streamlit
#pip install langchain
#pip install wikipedia
#pip install tiktoken
import os
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chains import LLMChain

from mykey import mykey
os.environ['OPENAI_API_KEY']=mykey

#interface design
st.title('Bella Playground')
#user provides input
prompt=st.text_input('Enter your desired cuisine')

#create few shot examples
examples = [
  {
    "question": "recommend me a restaurant in San Mateo that provides food similar to general chicken",
    "answer": 
"""
recommended restaurant: Ping's Bistro
customer rating: 3 stars
location: 2946 S Norfolk St, San Mateo, CA 94403, USA
healtiness index: 2 stars
"""
  }
]
example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")

#test if right format
#print(example_prompt.format(**examples[0]))

#create prompt template
merchant_prompt = FewShotPromptTemplate(
    examples=examples, 
    example_prompt=example_prompt, 
    suffix="Question: recommend me a restaurant in San Mateo that provides food similar to {input}", 
    input_variables=["input"]
)

#test if right format
#print(merchant_prompt.format(input="sushi"))

llm=OpenAI(temperature=0.6)
merchant_chain=LLMChain(llm=llm,prompt=merchant_prompt,verbose=True)

if prompt:
    response=merchant_chain.run(prompt)
    st.text(response)