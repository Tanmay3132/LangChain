#!/usr/bin/env python
# coding: utf-8

# In[92]:


pip install langchain


# In[93]:


import os
os.environ["OPENAI_API_KEY"] = "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"


# In[94]:


pip install openai


# In[95]:


from langchain.llms import OpenAI


# In[96]:


llm = OpenAI(temperature = 0.99)


# In[97]:


query = "can you list out 50 names for  a new born boy baby in india starting with 't'"
print(llm(query))


# In[98]:


query = "what is the current temperature in nagpur"
print(llm(query))


# In[99]:


query = "what is wheater in mumbai"
print(llm(query))


# In[100]:


query = "tell me a good joke"
print(llm(query))


# In[101]:


query = '''extract intent and entities from the below setence 
i want account Balance of account number 123456789 from 1 Dec 2021 to 31 Jan 2022
'''
print(llm(query))


# In[102]:


query = '''extract intent and entities from the below setence 
book a flight from nagpur NGP to delhi DEL on 27 march
'''
print(llm(query))


# In[103]:


query = '''extract intent and entities from the below setence 
block a credit card 12345467
'''
print(llm(query))


# # Chat Modal langChain 
# 
# 

# In[104]:


from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


# In[105]:


chat = ChatOpenAI(temperature=0)


# In[106]:


chat([HumanMessage(content="Translate this sentence from English to hindi. I love programming.")])


# In[113]:


messages = [
    SystemMessage(content="You are a helpful assistant that translates English to hindi."),
    HumanMessage(content="Translate this sentence from English to hindi. I love programming.")
]
resp = chat(messages)
print("\n" + resp.content + "\n")


# In[225]:


from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
query = "what is node js"
resp = chat([HumanMessage(content=query)], )


# In[214]:


from langchain.memory import ConversationBufferMemory


# In[227]:


import json

from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict

history = ChatMessageHistory()

history.add_user_message(query)

history.add_ai_message(resp.content)


# In[228]:


dicts = list()
dicts.append(messages_to_dict(history.messages))
context = dicts[-1][0]["data"]["content"] 
print(dicts[-1][0]["data"]["content"])


# In[229]:


memory = ConversationBufferMemory()
memory.chat_memory.add_user_message(query)
memory.chat_memory.add_ai_message(resp.content)


# In[230]:


previous_content = memory.load_memory_variables({})["history"].split("\n")[0].split("Human: ")[1]
print(previous_content)


# In[234]:


chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
query = context+ "," +"who created it"

resp = chat([HumanMessage(content=query)], memory=ConversationBufferMemory(),)

history.add_user_message(query)
history.add_ai_message(resp.content)
dicts.append(messages_to_dict(history.messages))

memory.chat_memory.add_user_message(query)
memory.chat_memory.add_ai_message(resp.content)


# In[232]:


print(dicts)
print("\n\n\n")
# context = dicts[-1][0]["data"]["content"] 
print(dicts[-1])
print("\n\n\n")
previous_content = memory.load_memory_variables(set())
print(previous_content)


# In[146]:


from langchain.llms import OpenAI
from langchain.chains import ConversationChain


llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)
conversation.predict(input="Hi there!")


# In[147]:


conversation.predict(input="what is node js")


# In[148]:


conversation.predict(input="when was it created")


# In[150]:


conversation.predict(input="where was he born")


# In[151]:


conversation.predict(input="from which university does he studied from ")

