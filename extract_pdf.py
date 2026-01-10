
import sys

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf
    except ImportError:
        print("MISSING_LIB")
        sys.exit(1)

try:
    reader = pypdf.PdfReader("instructions.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open("instructions.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Done")
except Exception as e:
    print(f"Error: {e}")
