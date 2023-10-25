import pdfplumber

def extract_text(pdf=None):
    # extract text from pdf
    if pdf is not None:
        text = ""
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

if __name__ == '__main__':
    print(extract_text("./sample_pdf/TheLittlePrince.pdf"))


