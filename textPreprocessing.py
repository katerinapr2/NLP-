import unicodedata

# 1. separate to sentences
# 2. edit text (remove punctuation, strip accents and lowercase)

class Preprocessing:

    def separate_to_sentences(text_file) -> str:
        text_str = ""
        with open(text_file, "r") as text:
            t = text.read()
            for i in range(0,len(t)):
                if (t[i] != "\n"):
                    if t[i] == '.':
                        text_str += t[i] + "\n"
                    else:
                        if t[i-1] + t[i] == ". ":
                            continue
                        else:
                            text_str += t[i]
        return text_str

    def strip_accents_and_lowercase(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', s)       #Normal Form Decomposed
                    if unicodedata.category(c) != 'Mn').lower()      #Mark, Nonspacing (τόνος)



    # def remove_punctuation(str_for_edit: str, write_file):
    #     punc = '''!()-[]}{;:'"\,<>./?@#$%^&*_~«»'''

    #     with open(write_file, "w") as without_punct:
    #         for character in str_for_edit:
    #             if character not in punc:
    #                 without_punct.write(character)
    #             else:
    #                 without_punct.write(" ")



