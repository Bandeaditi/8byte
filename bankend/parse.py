import pytesseract
from pdf2image import convert_from_path
from bankend.llm import parse_receipt_with_llm, categorize_vendor
from bankend.database import Receipt

def extract_text_from_image(image_path: str) -> str:
    return pytesseract.image_to_string(image_path)

def parse_receipt(file_path: str) -> Receipt:
    if file_path.endswith(".pdf"):
        images = convert_from_path(file_path)
        text = extract_text_from_image(images[0])
    else:  # JPG/PNG/TXT
        with open(file_path, "r") as f:
            text = f.read() if file_path.endswith(".txt") else extract_text_from_image(file_path)
    
    # LLM Parsing
    data = parse_receipt_with_llm(text)
    category = categorize_vendor(data["vendor"])
    return Receipt(
        vendor=data["vendor"],
        date=data["date"],
        amount=data["amount"],
        category=category
    )