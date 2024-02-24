from openai import OpenAI
# 设置 API key 和 API base URL
api_key = 'sk-NRdDBitwMqURITWb12B860Ad1b8446Fc9d71Cb5a488bF86a'
# api_key = 'sk-9tfSKYI2TLhfDXTG9bAf91A8DbD3497bA715F1111f7dFd5c'
base_url = "https://api.132999.xyz/v1"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))


def chatgpt(content):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    response = chatgpt('hello')
    print(response)