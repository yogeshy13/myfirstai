from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain_ollama import OllamaLLM

llm=OllamaLLM(model="llama3.1")

#examples=[{"sentance":"Hello","translation":"Hola"},
#          {"sentance":"Goodbye","translation":"Adios"}]

examples=[{"sentance":"Cricket","translation":"Bat"},
          {"sentance":"Soccer","translation":"Ball"}]

example_prompt = PromptTemplate.from_template("sentance: {sentance}\ntranslation: {translation}")

few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Translate the following phrases:",
        suffix="sentance: {query}\ntranslation:",
        input_variables=["query"]
    )

formatted_prompt = few_shot_prompt.format(query="Translate 'Hockey' to Type.")
print(formatted_prompt)
response=llm.invoke(formatted_prompt)
print(response)
