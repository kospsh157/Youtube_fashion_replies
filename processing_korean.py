from konlpy.tag import Okt
import konlpy
import jpype

# JVM 경로를 직접 지정
jvm_path = "/Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home/lib/server/libjvm.dylib"
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, '-Dfile.encoding=UTF8')

okt = Okt()


def main():
    okt = Okt()
    text = "안녕하세요! 자연어 처리를 공부하고 있습니다."

    # 토큰화
    tokens = okt.morphs(text)
    print("Tokens:", tokens)

    # 명사 추출
    nouns = okt.nouns(text)
    print("Nouns:", nouns)

    # 형태소 분석
    pos = okt.pos(text)
    print("Part-of-speech:", pos)


if __name__ == "__main__":
    main()
