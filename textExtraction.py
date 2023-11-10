import pdfplumber

def extract_text(pdf=None):
    with pdfplumber.open(pdf) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    with open("sample_pdf/output.txt", "w") as f:
        f.write(text)


if __name__ == '__main__':
    print(extract_text("./sample_pdf/TheLittlePrince.pdf"))


