from textExtraction import extract_text
from textSplit import split_text

print(split_text(extract_text("./sample_pdf/TheLittlePrince.pdf")))