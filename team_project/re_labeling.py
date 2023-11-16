import os

import matplotlib.pyplot as plt
from PIL import Image


dog_emotion_image_paths=[
    "./Dog Emotion/sad/6RocDWbOIYax0cS4OTu0oQ2HCmcuv5288.jpg",
    "./Dog Emotion/relaxed/81QP4kXADPl2FXcAmolI9OO2zoJb8D366.jpg",
    "./Dog Emotion/sad/5VvHcHATzGkqpzNiXO5e8B83pkKcwh323.jpg",
    "./Dog Emotion/sad/2qn0xIS8aqy5jyew88WAPwvUGOASAU782.jpg",
    "./Dog Emotion/relaxed/5xeREOu2rxbRbUnEospLpCyvxrLP2R3.jpg",
    "./Dog Emotion/sad/6FJANWlbGLt0XCx20ve17RUnUZeW6S590.jpg",
    "./Dog Emotion/relaxed/81asEVmpsVxwX7yCnU3DtN5OiaRUd5799.jpg",
    "./Dog Emotion/relaxed/4CUGFwEnHw9FwuvqCAn5agpjZ9ziYv307.jpg",
    "./Dog Emotion/angry/0da4j6Ehkb6Ml0YBRiWmsBU2wEMoXP409.jpg",
    "./Dog Emotion/relaxed/3bwvSv6deahWvCWIqRt8zeyWhO4GlE448.jpg",
    "./Dog Emotion/happy/26aBvOWVhZxMSoRVmSIrZaRxUl1674741.jpg",
    "./Dog Emotion/happy/81abPg9epvVT46sTl0MudaEejz7YPC891.jpg",
    "./Dog Emotion/happy/4GmFhvfxYTCz4XgynDBcWiPrQmk2cr108.jpg",
    "./Dog Emotion/happy/8SmgZZqiQ6troZHgPnONXgynxgk5zh596.jpg",
    "./Dog Emotion/sad/7p0wm7G6F4QUDYYEfg4PYeRFyBNSBc293.jpg",
    "./Dog Emotion/relaxed/BNS1tplaOIWScSBfSxG3E6mQzsgOZ9741.jpg",
    "./Dog Emotion/sad/8FM9t1TewJ0BstPBwju5NfCv8otgVy560.jpg",
    "./Dog Emotion/angry/BcPBwIMppWvhNyC4T4YqWVKtP7GtjH702.jpg",
    "./Dog Emotion/angry/62jTtCF0xpxlcsEtWt6qgQDK6aEyqZ229.jpg",
    "./Dog Emotion/sad/AWkHoxzBLKgQh9nh1l7soBKYlMTmsa198.jpg",
    "./Dog Emotion/angry/8tiQoVNWOx3ygVruJoCMux0Gopp1JN827.jpg",
    "./Dog Emotion/angry/8npDb0wmbM4M3hdXavCWLgDXcQFGJ2160.jpg",
    "./Dog Emotion/happy/5TwsL07giF0CYBladvu4Op680SE049795.jpg",
    "./Dog Emotion/sad/87PgHZO1gUR6BShuNoVPV5q3BT8Qru535.jpg",
    "./Dog Emotion/angry/6s1xCH6WkJevU2YuNdue5i7NrO1dEH587.jpg",
    "./Dog Emotion/angry/3gNqRVAOOxGGxL64hG2fXa2apwIDio797.jpg",
    "./Dog Emotion/angry/9HcHkBJYvQkMtwHOHXkT2lmfj1FCaE926.jpg",
    "./Dog Emotion/happy/5yAlsv3IimyvT8C0gAgXXNU1uBSQnZ585.jpg",
    "./Dog Emotion/sad/6hfYckJMO9XkVcV0jM77H7RcS25J3p972.jpg",
    "./Dog Emotion/angry/6gXpX9sOFx5FPp8W29lde9rwdhr239280.jpg",
    "./Dog Emotion/happy/70zKm35WFPLrECn5Y5meeTyvLVnCTx610.jpg",
    "./Dog Emotion/happy/3kjfQ3dDjZvjQYNVuheVcUgz6IRzTE704.jpg",
    "./Dog Emotion/sad/ATpwWSngW2sa9bYuf84sjaWgU4z9CX443.jpg",
    "./Dog Emotion/relaxed/28AWLqxSHxuWb8jLf9LESvQAlwcZEt655.jpg",
    "./Dog Emotion/sad/9cwsljcV9ub5XTiJk3ZB2d9j5grPkb13.jpg",
    "./Dog Emotion/sad/AcqIMIe0EO85n5kad3LIgRWPOH23i6330.jpg",
    "./Dog Emotion/sad/5zNmNnqwcJKWKLcRRdODiW6NvRf29A226.jpg",
    "./Dog Emotion/sad/1MpwUJ9k2oN6BRmFiONKNsoaNeJSph109.jpg",
    "./Dog Emotion/sad/0zdKmMEVmO8z6kvHXiTb8bGaqvhmgS278.jpg",
    "./Dog Emotion/angry/CDhGC09FlzKD7BmbUEj3tng0dl5mU2140.jpg",
    "./Dog Emotion/angry/5g5AGokiT61EY835V5hVux1DPsrMRu868.jpg",
]

for image_path in dog_emotion_image_paths: 
    img = Image.open(image_path)
    image_name = os.path.basename(image_path)
    emotion = os.path.basename(os.path.dirname(image_path))
    
    plt.figure()
    plt.imshow(img)
    plt.axis('off')  # 이미지 경계선 제거
    
    # 이미지 위에 이미지 이름 및 감정 표시
    plt.text(10, -20, f"Emotion: {emotion}%d", fontsize=12, color='black', backgroundcolor='white')
    plt.text(10, -40, f"Image Name: {image_name}", fontsize=12, color='black', backgroundcolor='white')
    
    plt.show()