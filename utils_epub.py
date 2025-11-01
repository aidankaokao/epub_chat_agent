from ebooklib import epub
from bs4 import BeautifulSoup

def parse_epub(epub_path: str):
    """解析 EPUB 取得純文字章節"""
    book = epub.read_epub(epub_path)
    texts = []
    for item in book.get_items():
        if item.get_type() == 9:  # ebooklib.ITEM_DOCUMENT
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text().strip()
            if text:
                texts.append(text)
    return texts