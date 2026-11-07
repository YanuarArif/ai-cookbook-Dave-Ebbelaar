from openai import OpenAI

client = OpenAI()

"""
https://platform.openai.com/docs/guides/reasoning?api-mode=responses
"""

prompt = """
Write a bash script that takes a matrix represented as a string with 
format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
"""

response = client.responses.create(
    model="deepseek-flash",
    reasoning={"effort": "medium"},
    input=[{"role": "user", "content": prompt}],
)

print(response.output_text)
