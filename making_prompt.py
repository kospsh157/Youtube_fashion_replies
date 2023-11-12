from openai import OpenAI

# 한글로 약 1000개의 글자당, 0.12불 요금

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)


def making_prompt(fashion_items):

    res = client.completions.create(
        model="text-davinci-003",
        prompt=f"Visualize an adult model in a sophisticated and stylish outfit, corresponding to the gender associated with the fashion items: {fashion_items}. The model should epitomize attractiveness and charm, similar to professional models. Ensure the outfit is tastefully chosen without duplication, reflecting a harmonious style and color palette. The model's portrayal should be dignified and elegant, avoiding any explicit or inappropriate content, and should present a realistic and appealing representation of fashion. The full outfit needs to be visible in a full-body depiction, suitable for all audiences.",
        max_tokens=300,
        temperature=0.6
    )

    print(res)
    print(res.choices[0].text.strip())
    return res.choices[0].text.strip()
