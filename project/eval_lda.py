from gensim.models import CoherenceModel


def eval_lda(texts, corpus, lda_model, dictionary):

    print('일단 EVal_lda 함수 실행됨....')
    # Coherence Model 생성 및 일관성 점수 계산
    coherence_model = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    print('Coherence Score:', coherence_score)

    # 퍼플렉서티 계산
    perplexity_score = lda_model.log_perplexity(corpus)

    print('Perplexity:', perplexity_score)

    return coherence_score, perplexity_score
