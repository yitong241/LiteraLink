import pdfplumber


def extract_text(pdf=None, start_page=6, end_page=8):
    with pdfplumber.open(pdf) as pdf:
        text = ''
        for i in range(start_page - 1, end_page):  
            page = pdf.pages[i]
            text += page.extract_text() + "\n"

    with open("sample_pdf/output.txt", "w") as f:
        f.write(text)
        
    return True


if __name__ == '__main__':
    print(extract_text("./sample_pdf/TheLittlePrince.pdf"))


