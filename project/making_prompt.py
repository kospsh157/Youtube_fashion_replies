from openai import OpenAI

# 한글로 약 1000개의 글자당, 0.12불 요금


from config import OPEN_API


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=OPEN_API
)


def making_prompt(tops, bottoms, outers, shoes, accessory):
    res1 = client.completions.create(
        model="text-davinci-003",
        prompt=f"I will provide you with five fashion items: a top ({tops}), bottom ({bottoms}), outerwear ({outers}), shoes ({shoes}), and accessories ({accessory}). \
            Please select one item from each category and devise a contemporary, appealing ensemble with a keen focus on color coordination. \
                Additionally, based on the chosen fashion items, decide whether a male or female model would best showcase the outfit. \
                    Your final response should be in English, detailing the model's gender, \
                        and the selected items in the order of top (color, material), bottom (color, material), outerwear (color, material), shoes (color, material), and accessory.",
        max_tokens=200,
        temperature=0.5
    )

    result = res1.choices[0].text.strip()
    print(result)

    # result 에서 패션 아이템들이 스타일링되고 prompt_2에서 하드코딩된 프롬프트에 전달된다. 그리고 이대로 이미지봇에 전달된다.
    prompt_2 = f"Please provide a full-length image of a professional fashion model featuring the specified items: {result}, \
        set against an outdoor backdrop. The model should be captured from head to toe,\
          including clearly visible footwear as it's an integral part of the fashion items. \
            Ensure the model is depicted with a natural human figure, specifically with two arms, \
                to avoid any unnatural representations. The setting should complement the fashion, \
                    with the model embodying the essence of modern style in the outdoor environment."
    # res2 = client.completions.create(
    #     model="text-davinci-003",
    #     prompt=f"Visualize an adult model in a sophisticated and stylish outfit, corresponding to the gender associated with the fashion items: {result}. \
    #         The model should epitomize attractiveness and charm, similar to professional models. \
    #             Ensure the outfit is tastefully chosen without duplication, reflecting a harmonious style and color palette. \
    #                 The model's portrayal should be dignified and elegant, avoiding any explicit or inappropriate content, and should present a realistic and appealing representation of fashion. \
    #                     The full outfit needs to be visible in a full-body depiction(including shoes), suitable for all audiences.",
    #     max_tokens=300,
    #     temperature=0.5
    # )

    # print(res)
    return prompt_2
    # return res2.choices[0].text.strip()
