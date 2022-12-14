import spacy


def semantic_similarity(keywords_list, current_keyword):
    # word want to test
    cur_keywords_list = keywords_list
    cur_keywords_list.append(current_keyword)
    cur_str = ""
    # construct food str
    for each in cur_keywords_list:
        cur_str += each
        cur_str += " "
    # print(food_str)
    tokens = nlp(food_str)
    token1, token2 = tokens[0:len(tokens) - 1], tokens[len(tokens) - 1]
    return token1.similarity(token2)


