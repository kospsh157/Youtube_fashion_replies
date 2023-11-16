from openai import OpenAI

# 한글로 약 1000개의 글자당, 0.12불 요금

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)


def making_prompt(tops, bottoms, outers, shoes, accessory):
    res1 = client.completions.create(
        model="text-davinci-003",
        prompt=f'Im going to provide you with five fashion items: a top({tops}), bottom({bottoms}), outerwear({outers}), shoes({shoes}), and accessories({accessory}). Please choose one from each category and create a modern, attractive styling with careful attention to color coordination. 최종결과는 패션 아이템 목록이어야해 (색상,재질)상의, (색상,재질)하의, (색상,재질)아우터, (색상,재질)신발, 악세사리 순서로. 단, 영어로 답변해줘.',
        max_tokens=200,
        temperature=0.5
    )

    result = res1.choices[0].text.strip()
    print(result)

    res2 = client.completions.create(
        model="text-davinci-003",
        prompt=f"Visualize an adult model in a sophisticated and stylish outfit, corresponding to the gender associated with the fashion items: {result}. The model should epitomize attractiveness and charm, similar to professional models. Ensure the outfit is tastefully chosen without duplication, reflecting a harmonious style and color palette. The model's portrayal should be dignified and elegant, avoiding any explicit or inappropriate content, and should present a realistic and appealing representation of fashion. The full outfit needs to be visible in a full-body depiction(including shoes), suitable for all audiences.",
        max_tokens=300,
        temperature=0.5
    )

    # print(res)
    # print(res.choices[0].text.strip())
    return res2.choices[0].text.strip()
