import pdfplumber


def extract_text(pdf=None, start_page=5, end_page=5):
    with pdfplumber.open(pdf) as pdf:
        total_pages = len(pdf.pages)
        text = ''
        for i in range(start_page - 1, end_page):  
            page = pdf.pages[i]
            text += page.extract_text() + "\n"

    # with open("sample_pdf/output.txt", "w") as f:
    #     f.write(text)

    # print("-----------------")
    # print("Total number of words selected:", len(text))
    # print("-----------------")
    return text, total_pages


if __name__ == '__main__':
    print(extract_text("./sample_pdf/Animal Farm.pdf"))
