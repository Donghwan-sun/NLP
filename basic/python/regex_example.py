import re

text = "my telephone number is 443-443-3343"

pattern = r"(\d{3})-(\d{3})-(\d{4})"
result = re.search(pattern, text)
print(result.group())

str_text = "This man was here"
pattern = r"man|woman"
print(re.search(pattern, str_text))
# . : 와일드카드를 의미
print(re.findall(r".at", "The cat in the hat sat"))

# ^ : "시작하는 글자가"를 의미 $: "끝나는 글자 "
print(re.findall(r"\d$", 'This ends with a number 2'))
print(re.findall(r"^\d", '1 is the loneliest number'))
phrase = "there are 3 numbers 34 inside 5 this sentence"
print(re.findall(r"[^\d]+",phrase))

test_phrase = "This is a string! but it has puncutation. Howto remove it?"
mylist = re.findall(r"[^\!.? ]+",test_phrase)

my_text= ' '.join(mylist)
print(my_text)



text = "Only find the hypthe-words. Were are the long-ish dash words?"

print(re.findall(r"[\w]+-[\w]+", text))