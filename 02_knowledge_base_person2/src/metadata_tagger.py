"""
Department classification and metadata tagging.

The mapping below was built by reviewing the FULL list of 171 unique
categories found in the dataset (see category_report.txt). Only
categories that clearly belong to one of our 3 target departments are
included below - everything else is intentionally left unmatched.

This module only exports functions - it doesn't run a pipeline by itself.
"""

# Maps a category name (exactly as it appears in the dataset's
# "categories" field) to one of our 3 target departments.
CATEGORY_TO_DEPARTMENT = {
    # --- Civil Affairs ---
    "أحوال شخصية للمسلمين": "civil_affairs",
    "أحوال شخصية غير المسلمين": "civil_affairs",
    "أحوال شخصية لغير المسلمين": "civil_affairs",
    "قوانين الأحوال الشخصية": "civil_affairs",
    "الاحوال الشخصية": "civil_affairs",
    "إرث": "civil_affairs",
    "المواريث": "civil_affairs",
    "جنسية": "civil_affairs",
    # "نقض مدني جزء اول" (1293 laws) was intentionally excluded: it's
    # general civil COURT RULINGS (contracts, rent, compensation),
    # not civil-registry matters (ID cards, birth certificates) that
    # our "Civil Affairs" department is meant to answer questions about.

    # --- Tax ---
    "ضرائب": "tax",
    "قوانين الضرائب": "tax",
    "ضريبة المبيعات": "tax",
    "ضريبة الدمغة": "tax",
    "قانون الضريبة على العقارات المبنية الجديد رقم 196 لسنة 2008": "tax",
    "جمارك": "tax",
    "رسوم": "tax",
    "رسم تنمية الموارد": "tax",

    # --- Traffic ---
    # Only one category in the whole dataset maps clearly to traffic.
    "قانون المرور": "traffic",
}

# Fallback keyword search on the law name, used only when the category
# didn't match anything above. Kept intentionally small and conservative.
LAW_NAME_KEYWORDS = {
    "civil_affairs": ["الاحوال الشخصيه", "الجنسيه", "شهاده الميلاد", "بطاقه الرقم القومي"],
    "tax": ["ضريبه", "ضرائب", "جمارك"],
    "traffic": ["مرور", "رخصه القياده", "المركبات"],
}


def get_department(law_name: str, categories: list) -> str | None:
    """
    Returns "civil_affairs", "tax", or "traffic" if the law belongs to
    one of our 3 target departments, otherwise returns None.

    Note: expects normalized text (see normalizer.py) for law_name,
    since LAW_NAME_KEYWORDS uses normalized spelling (e.g. "ه" instead
    of "ة"). The categories field is matched against its original,
    non-normalized form since that's how it appears in the dataset.
    """
    for category in categories:
        if category in CATEGORY_TO_DEPARTMENT:
            return CATEGORY_TO_DEPARTMENT[category]

    for department, keywords in LAW_NAME_KEYWORDS.items():
        for keyword in keywords:
            if keyword in law_name:
                return department

    return None


def build_metadata(law_name_original: str, categories: list,
                    department: str, chunk_index: int) -> dict:
    """
    Builds the metadata dictionary attached to every chunk.
    """
    return {
        "department": department,
        "law_name": law_name_original,
        "categories": categories,
        "chunk_index": chunk_index,
        "source": "Egyptian Legal Corpus",
    }
