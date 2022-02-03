person = "jose"

print("my name is {}".format(person))

print("my name is {person}")
print(f"my name is {person}")

d = {'a':123, "b": 345}
d_list = list(d.keys())
print(f"my number is {d['a']}")
print(f"my number is {d_list}")

library = [('Author', 'Topic', 'Pages'), ('Twain', 'Rafting in water alone', 601), ('Feynman', 'Physics', 95), ('Hamilton', 'Mythology', 144)]

print(library)
# spaCy 사용시 가독성이 좋지 않아 형식화 하는 구문
for author, topic, pages in library:
    print(f"Author is {author:{10}} {topic:30} {str(pages):{10}}")

from datetime import datetime
#strftime.org에서 %B, $d등 포맷을 찾아볼수있음
today = datetime(year=2019, month=2, day=28)
print(f"{today:%B %d, %Y}")