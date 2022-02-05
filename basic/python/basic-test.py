"""
    f-Strings
    1. Print an f-string that displays NLP stands for Natural Language Processing using the variables provided
    result = 'NLP Natural Language Processing'
"""
#1번 정답
abbr = 'NLP'
full_text = 'Natural Language Processing'

print(abbr + ' ' + full_text)
print(' '.join([abbr, full_text]))


"""
    Files
    2. Create a file in the current working directory called contacts.txt by running the cell below
    3. Open the file and use .read() to save the contents of the file to a string called fields. Make sure the file is closed at the end.
"""
# 2번 정답
with open('contacts.txt', 'w') as f:
    f.write("First_Name Last_Name, Title, Extension, Email")
# 3번 정답
with open('contacts.txt', 'r') as f:
    fields = f.read()
    
print(fields)

"""
    4. Use PyPDF2 to open the file Business_Proposal.pdf. Extract the text of page 2.
"""
# 4번 정답
import PyPDF2
pdf_path = 'C:\\project\\2022\\study\\NLP 강의자료\\UPDATED_NLP_COURSE\\UPDATED_NLP_COURSE\\00-Python-Text-Basics\\Business_Proposal.pdf'

pdf_content = open(pdf_path, mode="rb")

pdf = PyPDF2.PdfFileReader(pdf_content)
pdf_content_page = pdf.getPage(1)
page_two_text = pdf_content_page.extractText()
print(page_two_text)
print(page_two_text)

"""
    5. Open the file contacts.txt in append mode. Add the text of page 2 from above to contacts.txt.
        CHALLENGE: See if you can remove the word "AUTHORS:
"""
# 5번 정답
with open("contacts.txt", "a") as f:
    f.write(page_two_text)

# 5번 챌린지 정답
with open("contacts.txt", "a") as f:
    page_two_text = page_two_text.strip("AUTHORS:")
    f.write(page_two_text)
pdf_content.close()

"""
    6. Using the page_two_text variable created above, extract any email addresses that were contained in the file Business_Proposal.pdf.
"""
# 6번 정답
import re
pattern = r"[\w]+@[\w]+"
print(re.findall(pattern, page_two_text))