"""Read-only audit of Lighter market metadata for order-size constraints."""
import json
import os
from backtest import get_json

def main():
    data = get_json("/orderBookDetails", {"filter": "perp"})
    rows = []
    if isinstance(data, dict):
        for key in ("order_book_details", "orderBookDetails", "order_books", "orderBooks", "markets"):
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
            if isinstance(value, dict):
                for nested in ("order_book_details", "order_books", "markets"):
                    if isinstance(value.get(nested), list):
                        rows = value[nested]
                        break
                if rows:
                    break
    output = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        relevant = {k: v for k, v in row.items()
                    if any(token in k.lower() for token in
                           ("min", "step", "tick", "size", "amount", "notional", "decimal", "symbol", "market_id", "status"))}
        output.append(relevant)
    os.makedirs("market-constraints-results", exist_ok=True)
    with open("market-constraints-results/market_constraints.json", "w", encoding="utf-8") as f:
        json.dump({"metadata_row_count": len(rows), "rows_with_constraint_fields": len(output),
                   "source": "/api/v1/orderBookDetails?filter=perp", "markets": output},
                  f, indent=2, ensure_ascii=False)
    print(f"Metadata rows: {len(rows)}; captured market metadata rows: {len(output)}")
    print("This audit reports raw venue fields; no minimum is inferred if the API does not expose it.")

if __name__ == "__main__":
    main()
