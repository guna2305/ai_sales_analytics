def guess_default(columns, keywords):
    cols_lower = [c.lower() for c in columns]
    for kw in keywords:
        for i, col in enumerate(cols_lower):
            if kw in col:
                return i
    return 0
