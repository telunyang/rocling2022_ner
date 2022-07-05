import opencc
converter = opencc.OpenCC('s2tw.json')
with open('./dataset/ccks2017.json', 'r', encoding="utf-8") as file:
    s_content = file.read()

t_content = converter.convert(s_content)
with open("./dataset/ccks2017_m.json", "w", encoding="utf-8") as file:
    file.write(t_content)