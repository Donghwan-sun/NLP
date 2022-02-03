with open('myfile.txt', 'w' ,encoding='utf8') as f:
    f.write('my first file')

# windows의 경우 \ 한개대신 \\ 두개를 입력합니다.
myfile = open("C:\\project\\2022\\Deeplearning\\study\\NLP\\basic\\python\\myfile.txt")
# macOS의 경우 /를 입력합니다.
#myfile = open("C:/project/2022/Deeplearning/study/NLP/basic")

print(myfile.read())
#두번쨰 read()를 호출했을때 빈칸이 뜨는이유는 처음 read()시 파일의 맨 끝을 통과한 다음 전체 파일을 문자열로 반환하기 때문에
print(myfile.read())
# seek(0)을 이용하여 파일의 인덱스 0으로 이동
myfile.seek(0)
print(myfile.read())
myfile.close()