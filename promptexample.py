from langchain import PromptTemplate
from langchain_ollama import OllamaLLM
prompt=PromptTemplate(template="What is the capital of {Country} ?",input_variable=["Country"])
filled_prompt=prompt.format(Country="DELHI")
print(filled_prompt)

llm=OllamaLLM(model="llama3.1")
response=llm.invoke(filled_prompt)
print(response)
