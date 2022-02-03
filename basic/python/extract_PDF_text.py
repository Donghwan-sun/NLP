import PyPDF2
# 텍스트를 추출하고 싶은 pdf가 있는 경로
PATH = "C:/project/2022/study/NLP 강의자료/UPDATED_NLP_COURSE/UPDATED_NLP_COURSE/00-Python-Text-Basics/Business_Proposal.pdf"

# pdf 파일 읽기
myfile = open(PATH, mode='rb')

# PyPDF 리더 객체로 변환
pdf_reader = PyPDF2.PdfFileReader(myfile)

# PDF의 페이지수 출력
print(pdf_reader.numPages)

# PDF의 1장 읽어오기 
page_one = pdf_reader.getPage(0)

# PDF의 1장의 텍스트 추출
page_one.extractText()
print(page_one.extractText())

myfile.close()

# 파이썬으로 pdf 편집은 맨끝장 뒤에 붙여넣기만 가능

f = open(PATH, 'rb')

pdf_reader = PyPDF2.PdfFileReader(f)

first_page = pdf_reader.getPage(0)
pdf_writer = PyPDF2.PdfFileWriter()
pdf_writer.addPage(first_page)
pdf_output = open('MY_Brand_New.pdf', 'wb')

pdf_writer.write(pdf_output)
pdf_output.close()
f.close()

brand_new = open('./MY_Brand_New.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(brand_new)
print(pdf_reader.numPages)
print(pdf_reader.getPage(0))