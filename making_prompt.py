from openai import OpenAI

# 한글로 약 1000개의 글자당, 0.12불 요금

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)


def making_prompt(fashion_items):

    res = client.completions.create(
        model="text-davinci-003",
        prompt=f"Here are some fashion-related keywords,{fashion_items}. I want you to use these keywords to create an image of a single model, fully dressed in these fashion items from head to toe, ensuring the entire outfit is visible and presented elegantly. The model should be an adult and the attire should be tasteful and appropriate, with no explicit or excessive exposure. If the styling is feminine, the model should be a woman; if the styling is masculine, the model should be a man. Please avoid featuring children or elderly models, and ensure the focus is on showcasing the fashion items respectfully.",
        max_tokens=250,
        temperature=0.5
    )
    print(res.choices[0].text.strip())
    return res.choices[0].text.strip()
