"""Read visible Excel worksheets as labeled text for retrieval."""

from io import BytesIO
from zipfile import ZipFile

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

MAX_ROWS = 20_000
MAX_COLUMNS = 100
MAX_EXPANDED_BYTES = 30 * 1024 * 1024


def extract_sheets(content):
    with ZipFile(BytesIO(content)) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded workbook exceeds the 30 MB classroom limit.")

    workbook = load_workbook(BytesIO(content), read_only=True, data_only=False, keep_links=False)
    sheets = []
    try:
        for sheet in workbook.worksheets:
            if sheet.sheet_state != "visible":
                continue
            if (sheet.max_row or 0) > MAX_ROWS or (sheet.max_column or 0) > MAX_COLUMNS:
                raise ValueError("Use worksheets with at most 20,000 rows and 100 columns.")
            rows = [f"Worksheet: {sheet.title}"]
            for number, cells in enumerate(sheet.iter_rows(), 1):
                if number > MAX_ROWS:
                    raise ValueError("Worksheet exceeds the 20,000-row classroom limit.")
                values = []
                for column, cell in enumerate(cells, 1):
                    if cell.value is None or cell.value == "":
                        continue
                    if column > MAX_COLUMNS:
                        raise ValueError("Worksheet exceeds the 100-column classroom limit.")
                    value = str(cell.value).strip()
                    if cell.data_type == "f":
                        value = f"Formula (not evaluated): {value}"
                    values.append(f"{get_column_letter(column)}{number}: {value}")
                if values:
                    rows.append(f"Row {number} | " + " | ".join(values))
            if len(rows) > 1:
                sheets.append({
                    "page": None, "sheet": sheet.title,
                    "text": "\n".join(rows), "method": "Excel cells",
                })
    finally:
        workbook.close()
    if not sheets:
        raise ValueError("No readable cells were found in visible worksheets.")
    return sheets
